//! One product `M·x` in strips, a strip a host.
//!
//! The sequence and the solution of block Wiedemann are chains: a slice's
//! products follow one another, and a slice stays on one host for as long
//! as one host makes its products. A product is not a chain. Word `c` of
//! `M·x` is the sum of `x` over the relations that set column `c`, and no
//! word needs another, so the words over a range of the columns need the
//! entries in those columns and nothing else of the matrix. A [`Strip`]
//! holds those entries and makes those words, and the strips of one cut, in
//! their order, make the image end to end: the words one host makes, bit
//! for bit, since a sum over GF(2) has no order.
//!
//! A cut is into ranges of the columns' places in the image, of about as
//! many entries each. It follows from the matrix and the count of parts
//! alone, so every host cuts alike with nothing passed between them, and a
//! strip says what it is a part of in its [`StripPart`], which the host that
//! puts the image together reads before any product
//! ([`StripPart::are_cut_of`]).
//!
//! What crosses a product: that host sends each strip `x`, a word a
//! relation, and has back the strip's words, a word a column of its range.
//! For `R` relations, `C′` columns that some relation sets and `P` strips
//! that is `8·(P·R + C′)` bytes a product.

use core::ops::Range;
use std::sync::Arc;

use super::wiedemann::{matrix_fingerprint, Dimensions};
use super::{Block, Sparse, SparseMatrix};

/// What a strip is a part of, and which words of the image are its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StripPart {
    matrix: Dimensions,
    index: usize,
    of: usize,
    columns: Range<usize>,
}

impl StripPart {
    /// The matrix the strip was cut from.
    #[must_use]
    pub fn matrix(&self) -> &Dimensions {
        &self.matrix
    }

    /// Which part this is, of how many.
    #[must_use]
    pub fn index(&self) -> (usize, usize) {
        (self.index, self.of)
    }

    /// The places in the image of the words the strip makes: a range of
    /// the columns that some relation sets, as the strips order them, which
    /// is empty for a part past the last cut.
    #[must_use]
    pub fn columns(&self) -> Range<usize> {
        self.columns.clone()
    }

    /// Whether `parts`, in their order, are the whole of one cut of the
    /// product of `matrix`: every part of that matrix, part `i` of as many
    /// as there are in place `i`, and their ranges end to end from the
    /// first column to the last. What a host asks of the strips it is to
    /// put an image together from, before the first product: strips of
    /// another matrix or another cut, a part missing or one out of place
    /// are refused here and not found out by a wrong word.
    #[must_use]
    pub fn are_cut_of(parts: &[Self], matrix: &Dimensions) -> bool {
        let mut next = 0;
        for (index, part) in parts.iter().enumerate() {
            if part.matrix != *matrix
                || part.index != index
                || part.of != parts.len()
                || part.columns.start != next
                || part.columns.end < next
            {
                return false;
            }
            next = part.columns.end;
        }
        !parts.is_empty() && next == matrix.columns()
    }
}

/// One part of a cut of `M·x`: the entries of the matrix in a range of the
/// columns, and the words of the image over that range.
///
/// Of the matrix it holds four bytes an entry in its columns, the
/// relation's number, and eight bytes a column, where the column's list
/// begins; except that the densest columns of the whole matrix, those a
/// product on one host holds as masks, are where they fall in its range a
/// bit a relation each, a byte a relation for every eight of them.
/// Beside the entries it holds the block it reads, a word a relation, and
/// its words of the image, a word a column, each copied once a product,
/// and its pool of threads.
///
/// While it is made it holds besides, each for a part of the making: the
/// places of the matrix's columns, four bytes a declared column and
/// thirty-six a column that some relation sets; its lists a second time,
/// as they are laid flat, with twenty-four bytes a column of its range;
/// and eight bytes a relation. The matrix it is made from is read and not
/// kept.
pub struct Strip {
    part: StripPart,
    product: Sparse,
    /// The block a product reads, where the pool's threads read it.
    x: Arc<Block>,
    /// The strip's words of the image, where the pool's threads write them.
    image: Arc<Block>,
}

impl core::fmt::Debug for Strip {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Strip").field("part", &self.part).finish()
    }
}

impl Strip {
    /// Part `part` of `of` of the product of `matrix`, its passes on at
    /// most `threads` threads. The cut is the same whatever the threads and
    /// wherever it is made. `None` for a part out of range, a matrix with
    /// no rows or no columns, or one whose rows or columns pass thirty-two
    /// bits, which the product's lists count them in.
    #[must_use]
    pub fn of(matrix: &SparseMatrix, part: usize, of: usize, threads: usize) -> Option<Self> {
        let (rows, declared) = (matrix.rows(), matrix.columns());
        if part >= of || rows.is_empty() || declared == 0 || !Sparse::indexes(rows.len(), declared)
        {
            return None;
        }
        let (product, columns, set) = Sparse::strip(rows, declared, (part, of), threads);
        let part = StripPart {
            matrix: Dimensions::new(rows.len(), set, matrix_fingerprint(matrix)),
            index: part,
            of,
            columns,
        };
        Some(Self {
            x: Block::zeroed(rows.len()),
            image: Block::zeroed(part.columns.len()),
            part,
            product,
        })
    }

    /// What the strip is a part of, and which words of the image it makes.
    #[must_use]
    pub fn part(&self) -> &StripPart {
        &self.part
    }

    /// The strip's words of `M·x`, in the order of [`StripPart::columns`], for
    /// `x` a word a relation; `None` when `x` is not as many words as the
    /// matrix has relations, before anything is read of it.
    #[must_use]
    pub fn apply(&mut self, x: &[u64]) -> Option<Vec<u64>> {
        if x.len() != self.part.matrix.relations() {
            return None;
        }
        self.x.write(0, x);
        self.product.forward(&self.x, &self.image);
        Some(self.image.words().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Entries;
    use super::*;

    /// Knuth's MMIX linear congruential generator (TAOCP vol. 2, §3.3.4),
    /// its high word.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 32
        }

        /// A word of two draws.
        fn word(&mut self) -> u64 {
            self.next() << 32 | self.next()
        }
    }

    /// How many of a matrix's first columns are drawn as small primes'
    /// are, column `c` once in `c + 2` relations: more than a product
    /// takes as masks, so that the heaviest are told from the heavy.
    const HEAVY: usize = 200;

    /// A matrix as a sieve's is, of `columns` declared columns of which
    /// every other is left empty: the first set columns heavy, as the
    /// small primes are, and `weight` more to a relation among the rest.
    fn sieve_like(rows: usize, columns: usize, weight: usize, seed: u64) -> SparseMatrix {
        let mut rng = Lcg(seed);
        let set = columns / 2;
        let heavy = HEAVY.min(set);
        let lists = (0..rows)
            .map(|_| {
                let mut row: Vec<u32> = (0..heavy)
                    .filter(|&column| rng.next().is_multiple_of(column as u64 + 2))
                    .map(|column| 2 * column as u32)
                    .collect();
                if set > heavy {
                    let light = (set - heavy) as u64;
                    row.extend((0..weight).map(|_| 2 * (heavy as u64 + rng.next() % light) as u32));
                }
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        SparseMatrix::new(columns, lists)
    }

    /// Every strip of the cut of `matrix` into `of`, which is checked to
    /// be one.
    fn cut(matrix: &SparseMatrix, of: usize, threads: usize) -> Vec<Strip> {
        let strips: Vec<Strip> = (0..of)
            .map(|part| Strip::of(matrix, part, of, threads).expect("a part of the cut"))
            .collect();
        let parts: Vec<StripPart> = strips.iter().map(|strip| strip.part().clone()).collect();
        assert!(StripPart::are_cut_of(&parts, &Dimensions::of(matrix)));
        strips
    }

    /// `M·x` as one host makes it.
    fn whole(matrix: &SparseMatrix, x: &[u64], threads: usize) -> Vec<u64> {
        let product = Sparse::forward_only(matrix.rows().to_vec(), matrix.columns(), threads);
        let image = Block::zeroed(product.columns());
        product.forward(&Block::of(x.iter().copied()), &image);
        image.words().collect()
    }

    /// The entries a strip holds: its lists' and its masks' bits.
    fn entries(strip: &Strip) -> usize {
        let Entries::Listed { lists, .. } = strip.product.entries().0 else {
            panic!("a strip's entries are listed");
        };
        let masked: u32 = strip
            .product
            .dense
            .masks
            .iter()
            .map(|byte| byte.count_ones())
            .sum();
        lists.indices.len() + masked as usize
    }

    /// The strips of a cut make the image one host makes, word for word,
    /// whatever the parts, on matrices with and without dense columns,
    /// with more columns than rows and with more parts than columns; and
    /// they hold each entry of the matrix once between them.
    #[test]
    fn the_strips_of_a_cut_make_the_whole_image() {
        /// The most parts a matrix is cut into here: more than the columns
        /// that some relation sets in the three smallest.
        const MOST: usize = 400;
        for (rows, columns, weight, seed) in [
            (3, 6, 2, 1),
            (40, 14, 3, 2),
            (300, 720, 4, 3),
            (700, 1_300, 8, 4),
            (2_500, 4_800, 12, 5),
        ] {
            let matrix = sieve_like(rows, columns, weight, seed);
            let total: usize = matrix.rows().iter().map(Vec::len).sum();
            let mut rng = Lcg(seed ^ 0xbeef);
            let x: Vec<u64> = (0..rows).map(|_| rng.word()).collect();
            let image = whole(&matrix, &x, 1);
            for of in [1, 2, 3, 5, 8, (columns + 3).min(MOST)] {
                let mut strips = cut(&matrix, of, 1);
                assert_eq!(
                    strips.iter().map(entries).sum::<usize>(),
                    total,
                    "{rows} by {columns} in {of}: an entry held twice, or by none"
                );
                let made: Vec<u64> = strips
                    .iter_mut()
                    .flat_map(|strip| strip.apply(&x).expect("a block of the relations"))
                    .collect();
                assert_eq!(made, image, "{rows} by {columns} in {of} strips");
            }
        }
    }

    /// The cases the test above is to meet are met, and a cut is by the
    /// entries: with eight parts two strips hold masks, one of them masks
    /// alone, and one holds lists alone; a part that ends before the dense
    /// columns has an eighth of the entries to within its longest list,
    /// which no cut divides; and with more parts than columns, parts are
    /// empty.
    #[test]
    fn a_cut_is_by_the_entries_and_goes_through_the_dense_columns() {
        const OF: usize = 8;
        let matrix = sieve_like(2_500, 4_800, 12, 5);
        let strips = cut(&matrix, OF, 1);
        let masked = |strip: &Strip| strip.product.dense.count;
        let listed = |strip: &Strip| strip.part.columns.len() - masked(strip);
        assert!(strips.iter().filter(|strip| masked(strip) > 0).count() >= 2);
        assert!(strips
            .iter()
            .any(|strip| masked(strip) == 0 && listed(strip) > 0));
        assert!(strips
            .iter()
            .any(|strip| masked(strip) > 0 && listed(strip) == 0));
        let total: usize = strips.iter().map(entries).sum();
        let set = Dimensions::of(&matrix).columns();
        let lists = set - super::super::Dense::count_for(set);
        let mut before = 0;
        for strip in strips.iter().filter(|strip| strip.part.columns.end < lists) {
            let Entries::Listed { lists, .. } = strip.product.entries().0 else {
                panic!("a strip's entries are listed");
            };
            let longest = lists.offsets.windows(2).map(|pair| pair[1] - pair[0]).max();
            let share = entries(strip);
            assert!(
                share.abs_diff(total / OF) <= longest.expect("a column") + 1,
                "{share} of {total}"
            );
            before += 1;
        }
        assert!(before >= OF / 2, "{before} parts before the dense columns");
        let small = sieve_like(40, 14, 3, 2);
        let strips = cut(&small, 17, 1);
        assert!(strips.iter().any(|strip| strip.part.columns.is_empty()));
        for mut strip in strips {
            let made = strip.apply(&[0; 40]).expect("a block of the relations");
            assert_eq!(made.len(), strip.part.columns.len());
        }
    }

    /// A cut and its words are the same on one thread and on four, the
    /// matrix large enough that a pool of four is kept.
    #[test]
    fn a_cut_is_the_same_on_any_threads() {
        const ROWS: usize = 4 * super::super::MINIMUM_FOLDS_PER_WORKER + 2 * 64;
        let matrix = sieve_like(ROWS, 2 * ROWS - 600, 6, 11);
        let mut rng = Lcg(0xfeed);
        let x: Vec<u64> = (0..ROWS).map(|_| rng.word()).collect();
        let image = whole(&matrix, &x, 1);
        for of in [1, 3] {
            let mut alone = cut(&matrix, of, 1);
            let mut four = crate::parallel::with_budget(4, || cut(&matrix, of, 4));
            for (one, four) in alone.iter_mut().zip(&mut four) {
                assert_eq!(one.part(), four.part());
                assert_eq!(one.apply(&x), four.apply(&x));
            }
            let made: Vec<u64> = four
                .iter_mut()
                .flat_map(|strip| strip.apply(&x).expect("a block of the relations"))
                .collect();
            assert_eq!(made, image);
        }
    }

    /// What is not a cut of the matrix is refused before a product:
    /// strips of another matrix, of another count of parts, a part missing
    /// or out of its place; a part out of range is not made; and a block
    /// that is not of the relations is not read.
    #[test]
    fn what_is_not_a_cut_is_refused() {
        let matrix = sieve_like(300, 720, 4, 3);
        let other = sieve_like(300, 720, 4, 33);
        let dimensions = Dimensions::of(&matrix);
        let parts = |matrix: &SparseMatrix, of: usize| -> Vec<StripPart> {
            (0..of)
                .map(|part| {
                    Strip::of(matrix, part, of, 1)
                        .expect("a part")
                        .part()
                        .clone()
                })
                .collect()
        };
        let three = parts(&matrix, 3);
        assert!(StripPart::are_cut_of(&three, &dimensions));
        assert!(!StripPart::are_cut_of(&parts(&other, 3), &dimensions));
        assert!(!StripPart::are_cut_of(&three, &Dimensions::of(&other)));
        assert!(!StripPart::are_cut_of(&three[..2], &dimensions));
        assert!(!StripPart::are_cut_of(&three[1..], &dimensions));
        assert!(!StripPart::are_cut_of(&[], &dimensions));
        let mut swapped = three.clone();
        swapped.swap(0, 1);
        assert!(!StripPart::are_cut_of(&swapped, &dimensions));
        let mut mixed = three.clone();
        mixed[1] = parts(&matrix, 4)[1].clone();
        assert!(!StripPart::are_cut_of(&mixed, &dimensions));
        let mut foreign = three.clone();
        foreign[2] = parts(&other, 3)[2].clone();
        assert!(!StripPart::are_cut_of(&foreign, &dimensions));

        assert!(Strip::of(&matrix, 3, 3, 1).is_none());
        assert!(Strip::of(&matrix, 0, 0, 1).is_none());
        assert!(Strip::of(&SparseMatrix::new(4, Vec::new()), 0, 1, 1).is_none());
        assert!(Strip::of(&SparseMatrix::new(0, vec![Vec::new()]), 0, 1, 1).is_none());
        let mut strip = Strip::of(&matrix, 1, 3, 1).expect("a part");
        assert_eq!(strip.apply(&[0; 299]), None);
        assert_eq!(strip.apply(&[0; 301]), None);
        assert!(strip.apply(&[0; 300]).is_some());
    }

    /// A matrix none of whose rows sets a column has no image, and its
    /// strips are empty.
    #[test]
    fn a_matrix_of_empty_rows_has_empty_strips() {
        let matrix = SparseMatrix::new(5, vec![Vec::new(), Vec::new()]);
        let mut strips = cut(&matrix, 2, 1);
        for strip in &mut strips {
            assert_eq!(strip.apply(&[1, 2]), Some(Vec::new()));
        }
    }
}
