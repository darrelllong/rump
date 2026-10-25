//! The generator of a block Wiedemann sequence by Thomé's recursion
//! (*Subquadratic computation of vector generating polynomials and
//! improvement of the block Wiedemann algorithm*, J. Symbolic Comput. 33
//! (2002), 757–775; his habilitation, ch. 12, Algorithm 12.3).
//!
//! Coppersmith's iteration keeps `A·F = G + Xᵗ·E` for `F` of `m + n`
//! columns, with a degree `δⱼ` a column bounding `F`'s and `1 + G`'s. A
//! step reads only `E`'s constant coefficient: it eliminates among the
//! columns in order of `δ`, a column receiving only columns of no greater
//! `δ`, and multiplies by `X` the `m` that stay non-zero. The step is a
//! matrix `π` of polynomials: `F ← F·π`, `E ← E·π div X`. The `π` of `k`
//! steps is fixed by `E`'s first `k` coefficients and `δ`, so it is made by
//! halves: the first half's from `E mod X^{k/2}`, the second's from
//! `(E·π_L mod X^k) div X^{k/2}`, the whole `π_L·π_R`. The products are of
//! polynomials whose coefficients are matrices, by Karatsuba's split. Where
//! each method gives way to the next, schoolbook to Karatsuba and steps one
//! at a time to halves, is worked out from what each costs at the shape in
//! hand ([`Shape::karatsuba_from`], [`halves_from`]).

use super::WIDTH;

// Costs are counted in word XORs streamed through memory, 0.165 ns each on
// an M4 Pro over slices of a few thousand words. The kernels here were
// timed at 1, 2, 4 and 8 words a column, 128 and 512 columns, on it and on
// a GB10's Grace cores, which came within a third of it (PERFORMANCE.md,
// "Thomé's recursion for the generator"); what each access costs beside
// its words is what sets the crossovers.

/// A table entry, beside its words: 1.05–1.64 ns an entry at 1–8 words, so
/// `6 + w/2` streamed XORs.
const ENTRY_FIXED: usize = 6;

/// A lookup, beside its words: 1.39–3.10 ns at 1–8 words, so `9 + w`
/// streamed XORs.
const LOOKUP_FIXED: usize = 9;

/// A slice summed into another, beside its words: 16-word slices stream at
/// 2.6·10⁹ words a second against 7·10⁹ for long ones, 3.4 ns a slice.
const SLICE_FIXED: usize = 20;

/// The streamed XORs a thread is given at the least. A scoped thread's
/// start and join cost 9–10 µs on the M4 Pro and 23–29 µs on the Grace
/// cores, at most 1.8·10⁵ streamed XORs; a share ten times that keeps the
/// threads' cost under a tenth of the work.
const GRAIN_XORS: usize = 1 << 21;

/// The threads worth giving `xors` of work: one a grain, within the budget.
fn threads_for(xors: usize) -> usize {
    (xors / GRAIN_XORS).clamp(1, crate::parallel::budget())
}

/// A matrix of polynomials over GF(2): `columns` columns of `words` words,
/// a bit a row, coefficient by coefficient and in each column by column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Polynomials {
    words: usize,
    columns: usize,
    data: Vec<u64>,
}

impl Polynomials {
    pub(super) fn zero(words: usize, columns: usize, length: usize) -> Self {
        Self {
            words,
            columns,
            data: vec![0; words * columns * length],
        }
    }

    /// The words of a coefficient.
    fn stride(&self) -> usize {
        self.words * self.columns
    }

    pub(super) fn length(&self) -> usize {
        self.data.len() / self.stride()
    }

    /// Column `column` of coefficient `index`.
    pub(super) fn column(&self, index: usize, column: usize) -> &[u64] {
        let at = index * self.stride() + column * self.words;
        &self.data[at..at + self.words]
    }

    pub(super) fn column_mut(&mut self, index: usize, column: usize) -> &mut [u64] {
        let at = index * self.stride() + column * self.words;
        &mut self.data[at..at + self.words]
    }

    /// Coefficients `range`, as a polynomial of their own.
    pub(super) fn coefficients(&self, range: std::ops::Range<usize>) -> Self {
        let stride = self.stride();
        Self {
            words: self.words,
            columns: self.columns,
            data: self.data[range.start * stride..range.end * stride].to_vec(),
        }
    }

    /// Without the zero coefficients at the top, one kept.
    pub(super) fn from_words(words: usize, columns: usize, data: Vec<u64>) -> Self {
        Self {
            words,
            columns,
            data,
        }
    }

    pub(super) fn words(&self) -> &[u64] {
        &self.data
    }

    /// Whether column `column` is zero in every coefficient.
    pub(super) fn zero_column(&self, column: usize) -> bool {
        (0..self.length()).all(|index| self.column(index, column).iter().all(|&word| word == 0))
    }

    pub(super) fn trimmed(mut self) -> Self {
        let stride = self.stride();
        let length = self
            .data
            .chunks_exact(stride)
            .rposition(|coefficient| coefficient.iter().any(|&word| word != 0))
            .map_or(1, |top| top + 1);
        self.data.truncate(length * stride);
        self
    }
}

/// The shape of a product `a·b`: `a`'s columns are `b`'s rows.
#[derive(Clone, Copy)]
struct Shape {
    /// `a`'s words a column, which are the product's.
    words: usize,
    /// `a`'s columns, `b`'s rows.
    inner: usize,
    /// `b`'s columns, the product's.
    columns: usize,
}

impl Shape {
    fn a(&self) -> usize {
        self.words * self.inner
    }

    fn b(&self) -> usize {
        self.inner / WIDTH * self.columns
    }

    fn out(&self) -> usize {
        self.words * self.columns
    }

    /// A coefficient's tables: 255 entries a byte of its columns.
    fn table(&self) -> usize {
        self.inner / 8 * 255 * (ENTRY_FIXED + self.words / 2)
    }

    /// A coefficient of `a`, by its tables, times one of `b`: a lookup a
    /// byte of each of `b`'s columns, at most.
    fn lookup(&self) -> usize {
        self.columns * self.inner / 8 * (LOOKUP_FIXED + self.words)
    }

    /// `n` coefficients a side by schoolbook: each of `a`'s tables made once
    /// and looked up with each of `b`'s coefficients.
    fn schoolbook(&self, n: usize) -> usize {
        n * self.table() + n * n * self.lookup()
    }

    /// The streamed XORs of one split of `n`, beside its three products: the
    /// two sums of halves, and the three products' sums into the whole, low
    /// and high twice and the middle once.
    fn split(&self, n: usize) -> usize {
        let h = n.div_ceil(2);
        h * (self.a() + self.b()) + (3 * (2 * h - 1) + 2 * (2 * (n - h) - 1)) * self.out()
    }

    /// The least `n` at which a split of schoolbook halves costs less than
    /// schoolbook. Splits are taken from there up: schoolbook's `n²` term
    /// grows faster than a split's.
    fn karatsuba_from(&self) -> usize {
        let mut cost = vec![0, self.schoolbook(1)];
        for n in 2usize.. {
            let split = 3 * cost[n.div_ceil(2)] + self.split(n);
            if split < self.schoolbook(n) {
                return n;
            }
            cost.push(self.schoolbook(n));
        }
        unreachable!("schoolbook outgrows a split")
    }

    /// The streamed XORs of a balanced product of `n` coefficients a side.
    fn balanced(&self, n: usize, from: usize) -> usize {
        if n < from {
            self.schoolbook(n)
        } else {
            3 * self.balanced(n.div_ceil(2), from) + self.split(n)
        }
    }

    /// The streamed XORs of a product of `la` coefficients by `lb`: whole
    /// pieces of the shorter's length, and the rest as a product of its own.
    fn product(&self, la: usize, lb: usize) -> usize {
        let (long, short) = (la.max(lb), la.min(lb));
        if short == 0 {
            return 0;
        }
        let rest = long % short;
        long / short * self.balanced(short, self.karatsuba_from()) + self.product(rest, short)
    }
}

/// `a·b`.
pub(super) fn product(a: &Polynomials, b: &Polynomials) -> Polynomials {
    assert_eq!(a.columns, b.words * WIDTH, "a's columns are b's rows");
    let shape = Shape {
        words: a.words,
        inner: a.columns,
        columns: b.columns,
    };
    let mut out = Polynomials::zero(shape.words, shape.columns, a.length() + b.length() - 1);
    product_into(&a.data, &b.data, &mut out.data, shape);
    out
}

/// `a·b` XORed into `out`, of `la + lb − 1` coefficients: whole pieces of the
/// shorter's length balanced, side by side, and the rest a product of its
/// own, the shorter cut in its turn.
fn product_into(a: &[u64], b: &[u64], out: &mut [u64], shape: Shape) {
    let (la, lb) = (a.len() / shape.a(), b.len() / shape.b());
    if la == 0 || lb == 0 {
        return;
    }
    let from = shape.karatsuba_from();
    let short = la.min(lb);
    let whole = la.max(lb) / short;
    let pieces: Vec<usize> = (0..whole).collect();
    let made = |_: usize, &index: &usize| {
        let mut made = vec![0u64; (2 * short - 1) * shape.out()];
        if la >= lb {
            karatsuba(
                &a[index * short * shape.a()..][..short * shape.a()],
                b,
                &mut made,
                shape,
                from,
            );
        } else {
            karatsuba(
                a,
                &b[index * short * shape.b()..][..short * shape.b()],
                &mut made,
                shape,
                from,
            );
        }
        made
    };
    let threads = threads_for(whole * shape.balanced(short, from));
    let made: Vec<Vec<u64>> = crate::parallel::map_ordered(&pieces, threads, made);
    for (index, made) in made.iter().enumerate() {
        xor(&mut out[index * short * shape.out()..][..made.len()], made);
    }
    let done = whole * short;
    let at = done * shape.out();
    if la >= lb {
        product_into(&a[done * shape.a()..], b, &mut out[at..], shape);
    } else {
        product_into(a, &b[done * shape.b()..], &mut out[at..], shape);
    }
}

/// `a·b` XORed into `out`, `a` and `b` of `n` coefficients each and `out`
/// of `2n − 1`: by Karatsuba's split from `from` coefficients, and below it
/// by schoolbook.
fn karatsuba(a: &[u64], b: &[u64], out: &mut [u64], shape: Shape, from: usize) {
    let n = a.len() / shape.a();
    if n < from {
        for (i, a) in a.chunks_exact(shape.a()).enumerate() {
            let tables = tables(a, shape);
            for (j, b) in b.chunks_exact(shape.b()).enumerate() {
                apply(
                    &tables,
                    b,
                    &mut out[(i + j) * shape.out()..][..shape.out()],
                    shape,
                );
            }
        }
        return;
    }
    let h = n.div_ceil(2);
    let (a0, a1) = a.split_at(h * shape.a());
    let (b0, b1) = b.split_at(h * shape.b());
    let mut a_sum = a0.to_vec();
    xor(&mut a_sum[..a1.len()], a1);
    let mut b_sum = b0.to_vec();
    xor(&mut b_sum[..b1.len()], b1);
    let halves: [(&[u64], &[u64]); 3] = [(a0, b0), (&a_sum, &b_sum), (a1, b1)];
    let half = |_: usize, &(x, y): &(&[u64], &[u64])| {
        let mut made = vec![0u64; (2 * (x.len() / shape.a()) - 1) * shape.out()];
        karatsuba(x, y, &mut made, shape, from);
        made
    };
    let made: Vec<Vec<u64>> =
        crate::parallel::map_ordered(&halves, threads_for(shape.balanced(n, from)), half);
    let (low, middle, high) = (&made[0], &made[1], &made[2]);
    // a·b = low + X^h·(middle − low − high) + X^{2h}·high.
    xor(&mut out[..low.len()], low);
    xor(&mut out[2 * h * shape.out()..][..high.len()], high);
    let at = h * shape.out();
    xor(&mut out[at..][..middle.len()], middle);
    xor(&mut out[at..][..low.len()], low);
    xor(&mut out[at..][..high.len()], high);
}

/// For each byte of a coefficient's columns, the XOR of the columns each of
/// its 256 selectors picks, `shape.words` words an entry: entries `2^i` to
/// `2^{i+1}` are entries `0` to `2^i` with column `i` added, a block a bit.
fn tables(coefficient: &[u64], shape: Shape) -> Vec<u64> {
    let words = shape.words;
    let mut tables = vec![0u64; shape.inner / 8 * 256 * words];
    for (table, columns) in tables
        .chunks_exact_mut(256 * words)
        .zip(coefficient.chunks_exact(8 * words))
    {
        for (i, column) in columns.chunks_exact(words).enumerate() {
            let (low, high) = table[..(2 << i) * words].split_at_mut((1 << i) * words);
            for (high, low) in high.chunks_exact_mut(words).zip(low.chunks_exact(words)) {
                for ((high, &low), &add) in high.iter_mut().zip(low).zip(column) {
                    *high = low ^ add;
                }
            }
        }
    }
    tables
}

/// A coefficient of `a`, by its tables, times one of `b`, XORed into one of
/// `out`: each column of `b` selects `a`'s columns a byte at a time.
fn apply(tables: &[u64], b: &[u64], out: &mut [u64], shape: Shape) {
    let words = shape.words;
    let inner = shape.inner / WIDTH;
    for (column, out) in b
        .chunks_exact(inner)
        .zip(out.chunks_exact_mut(words))
        .take(shape.columns)
    {
        for (j, &word) in column.iter().enumerate() {
            let mut value = word;
            let mut byte = j * 8;
            while value != 0 {
                let selector = (value & 0xff) as usize;
                if selector != 0 {
                    let at = (byte * 256 + selector) * words;
                    xor(out, &tables[at..at + words]);
                }
                value >>= 8;
                byte += 1;
            }
        }
    }
}

fn xor(into: &mut [u64], from: &[u64]) {
    for (word, &add) in into.iter_mut().zip(from) {
        *word ^= add;
    }
}

/// The iteration's state beside `π`: each column's `δ`, and the steps it has
/// gone without being multiplied by `X`.
pub(super) struct Degrees {
    pub(super) degrees: Vec<usize>,
    pub(super) quiet: Vec<usize>,
}

/// The streamed XORs of `k` steps one at a time on `count` columns of
/// `words` words. A column at place `p` of the order is reduced by each of
/// the pivots before it with probability one half, and there are `count/2`
/// pivots, so the columns receive `Σ_{p<count/2} p/2 + (count/2)·(count/4)
/// = 3·count²/16` columns a step. Each reduction is two slices, a term and
/// a set, `words + count/64`; each column received is two more, `E`'s
/// coefficients still to come and `π`'s so far, `(k − s)·words +
/// (s + 1)·count/64` at step `s`, `k(k + 1)/2·(words + count/64)` over
/// the steps.
fn one_at_a_time_xors(k: usize, count: usize, words: usize) -> usize {
    let set = count / WIDTH;
    3 * count * count / 16 * (k * (4 * SLICE_FIXED + words + set) + k * (k + 1) / 2 * (words + set))
}

/// The least number of steps at which halves cost less than taking them
/// one at a time, for `count` columns of `words` words: the halves' own
/// steps, `E·π_L` from where it reaches `X^j`, and `π_L·π_R`. Each half's
/// `π` has degree about half its steps, `m` of the `2m` columns multiplied
/// by `X` a step. The steps grow as `k²` and the halves as `k^{log₂3}`.
pub(super) fn halves_from(count: usize, words: usize) -> usize {
    let e = Shape {
        words,
        inner: count,
        columns: count,
    };
    let pi = Shape {
        words: count / WIDTH,
        inner: count,
        columns: count,
    };
    let mut cost = vec![0];
    for k in 1usize.. {
        let one = one_at_a_time_xors(k, count, words);
        if k >= 2 {
            let j = k / 2;
            let (lj, lk) = (j / 2 + 1, (k - j) / 2 + 1);
            let halves =
                cost[j] + cost[k - j] + e.product(k - (j + 1 - lj), lj) + pi.product(lj, lk);
            if halves < one {
                return k;
            }
        }
        cost.push(one);
    }
    unreachable!("the steps outgrow the halves")
}

/// The `π` of as many steps as `e` has coefficients, by halves from `from`
/// steps and one at a time below, and `state` advanced over them.
pub(super) fn steps(e: &Polynomials, state: &mut Degrees, from: usize) -> Polynomials {
    let k = e.length();
    if k < from.max(2) {
        return one_at_a_time(e, state);
    }
    let j = k / 2;
    let left = steps(&e.coefficients(0..j), state, from);
    // E·π_L is divisible by X^j and its next k − j coefficients are the
    // second half's E; E's coefficients below j − deg π_L reach none of them.
    let low = (j + 1).saturating_sub(left.length());
    let right_e = product(&e.coefficients(low..k), &left).coefficients(j - low..k - low);
    let right = steps(&right_e, state, from);
    product(&left, &right).trimmed()
}

/// The steps one at a time, on `e`'s columns each held whole: a step sums
/// columns of `E` and of `π`, and multiplies some by `X`.
fn one_at_a_time(e: &Polynomials, state: &mut Degrees) -> Polynomials {
    let (k, count, words) = (e.length(), e.columns, e.words);
    let set_words = count / WIDTH;
    // E column by column, k coefficients each; π likewise, k + 1.
    let (e_span, p_span) = (k * words, (k + 1) * set_words);
    let mut series = vec![0u64; count * e_span];
    for i in 0..k {
        for c in 0..count {
            series[c * e_span + i * words..][..words].copy_from_slice(e.column(i, c));
        }
    }
    let mut pi = vec![0u64; count * p_span];
    for c in 0..count {
        pi[c * p_span + c / WIDTH] = 1u64 << (c % WIDTH);
    }
    let mut next_series = vec![0u64; series.len()];
    let mut next_pi = vec![0u64; pi.len()];
    let mut terms = vec![0u64; count * words];
    let mut received = vec![0u64; count * set_words];
    let mut pivot = vec![false; count];
    let mut pivots: Vec<(usize, usize)> = Vec::with_capacity(count);
    for s in 0..k {
        // The elimination on the coefficients at s alone: which columns each
        // column receives, as a set of the columns the step began with.
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&c| (state.degrees[c], c));
        received.fill(0);
        pivots.clear();
        pivot.fill(false);
        for &c in &order {
            let mut term = series[c * e_span + s * words..][..words].to_vec();
            let mut set = vec![0u64; set_words];
            set[c / WIDTH] = 1u64 << (c % WIDTH);
            for &(bit, other) in &pivots {
                if term[bit / WIDTH] >> (bit % WIDTH) & 1 == 1 {
                    xor(&mut term, &terms[other * words..][..words]);
                    xor(&mut set, &received[other * set_words..][..set_words]);
                }
            }
            if let Some(bit) = term.iter().enumerate().find_map(|(w, &word)| {
                (word != 0).then(|| w * WIDTH + word.trailing_zeros() as usize)
            }) {
                pivots.push((bit, c));
                pivot[c] = true;
            }
            terms[c * words..][..words].copy_from_slice(&term);
            received[c * set_words..][..set_words].copy_from_slice(&set);
        }

        // The columns themselves: each the sum of those it receives, the
        // pivots then multiplied by X. Below s the coefficients of E are
        // zero and are not read again; π reaches s + 1.
        let (old_series, old_pi, received, pivot) = (&series, &pi, &received, &pivot);
        let column = |c: usize, e_out: &mut [u64], p_out: &mut [u64]| {
            let shift = usize::from(pivot[c]);
            e_out[s * words..].fill(0);
            p_out[..(s + 2) * set_words].fill(0);
            for (w, &word) in received[c * set_words..][..set_words].iter().enumerate() {
                let mut others = word;
                while others != 0 {
                    let o = w * WIDTH + others.trailing_zeros() as usize;
                    others &= others - 1;
                    let from = &old_series[o * e_span..][s * words..(k - shift) * words];
                    xor(&mut e_out[(s + shift) * words..], from);
                    let from = &old_pi[o * p_span..][..(s + 1) * set_words];
                    xor(&mut p_out[shift * set_words..], from);
                }
            }
        };
        let work = 3 * count * count / 16 * ((k - s) * words + (s + 1) * set_words);
        let threads = threads_for(work).min(count);
        if threads > 1 {
            let per = count.div_ceil(threads);
            std::thread::scope(|scope| {
                for (part, (e_part, p_part)) in next_series
                    .chunks_mut(per * e_span)
                    .zip(next_pi.chunks_mut(per * p_span))
                    .enumerate()
                {
                    let column = &column;
                    scope.spawn(move || {
                        for (index, (e_out, p_out)) in e_part
                            .chunks_mut(e_span)
                            .zip(p_part.chunks_mut(p_span))
                            .enumerate()
                        {
                            column(part * per + index, e_out, p_out);
                        }
                    });
                }
            });
        } else {
            for (c, (e_out, p_out)) in next_series
                .chunks_mut(e_span)
                .zip(next_pi.chunks_mut(p_span))
                .enumerate()
            {
                column(c, e_out, p_out);
            }
        }
        std::mem::swap(&mut series, &mut next_series);
        std::mem::swap(&mut pi, &mut next_pi);
        for ((&pivot, degree), quiet) in pivot.iter().zip(&mut state.degrees).zip(&mut state.quiet)
        {
            if pivot {
                *degree += 1;
                *quiet = 0;
            } else {
                *quiet += 1;
            }
        }
    }
    let mut out = Polynomials::zero(set_words, count, k + 1);
    for i in 0..=k {
        for c in 0..count {
            out.column_mut(i, c)
                .copy_from_slice(&pi[c * p_span + i * set_words..][..set_words]);
        }
    }
    out.trimmed()
}

/// The transpose of a 64 by 64 block of bits, bit `c` of word `r` to bit
/// `r` of word `c`: the off-diagonal halves exchanged, then the quarters
/// within each, down to single bits.
pub(super) fn transpose(block: &mut [u64; 64]) {
    let mut width = 32;
    let mut mask: u64 = 0x0000_0000_ffff_ffff;
    while width != 0 {
        let mut r = 0;
        while r < 64 {
            for i in r..r + width {
                let t = ((block[i] >> width) ^ block[i + width]) & mask;
                block[i] ^= t << width;
                block[i + width] ^= t;
            }
            r += 2 * width;
        }
        width >>= 1;
        mask ^= mask << width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 (Steele, Lea & Flood, *Fast splittable pseudorandom
    /// number generators*, OOPSLA 2014).
    struct Draw(u64);

    impl Draw {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn polynomials(&mut self, words: usize, columns: usize, length: usize) -> Polynomials {
            let mut p = Polynomials::zero(words, columns, length);
            for word in &mut p.data {
                *word = self.next();
            }
            p
        }
    }

    fn bit(p: &Polynomials, index: usize, row: usize, column: usize) -> bool {
        p.column(index, column)[row / WIDTH] >> (row % WIDTH) & 1 == 1
    }

    /// The product bit by bit, from the definitions.
    fn plain(a: &Polynomials, b: &Polynomials) -> Polynomials {
        let rows = a.words * WIDTH;
        let mut out = Polynomials::zero(a.words, b.columns, a.length() + b.length() - 1);
        for i in 0..a.length() {
            for j in 0..b.length() {
                for c in 0..b.columns {
                    for inner in 0..a.columns {
                        if bit(b, j, inner, c) {
                            for r in 0..rows {
                                if bit(a, i, r, inner) {
                                    out.column_mut(i + j, c)[r / WIDTH] ^= 1u64 << (r % WIDTH);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Balanced and not, odd lengths and even, either side the longer, at
    /// the shapes the recursion makes: `E` by `π` and `π` by `π`.
    #[test]
    fn the_product_is_the_plain_product() {
        let mut draw = Draw(3);
        for (la, lb) in [
            (1, 1),
            (2, 2),
            (3, 3),
            (5, 2),
            (2, 7),
            (8, 8),
            (9, 4),
            (13, 13),
        ] {
            for (words, of) in [(1usize, 1usize), (2, 1)] {
                let a = draw.polynomials(words, 2 * WIDTH * of, la);
                let b = draw.polynomials(2 * of, 2 * WIDTH * of, lb);
                assert_eq!(
                    product(&a, &b),
                    plain(&a, &b),
                    "{la} by {lb}, {words} words"
                );
            }
        }
    }

    #[test]
    fn transpose_moves_bit_c_of_word_r_to_bit_r_of_word_c() {
        let mut draw = Draw(9);
        let mut block = [0u64; 64];
        for word in &mut block {
            *word = draw.next();
        }
        let before = block;
        transpose(&mut block);
        for (r, &row) in before.iter().enumerate() {
            for (c, &column) in block.iter().enumerate() {
                assert_eq!(row >> c & 1, column >> r & 1, "({r}, {c})");
            }
        }
    }

    /// A series whose steps find their pivots: random `E` of `of` words and
    /// `2·64·of` columns.
    fn series(of: usize, length: usize, seed: u64) -> Polynomials {
        Draw(seed).polynomials(of, 2 * WIDTH * of, length)
    }

    /// Halves from every number of steps make the same `π`, degrees and
    /// quiet steps as all the steps one at a time: the recursion is the
    /// iteration.
    #[test]
    fn every_leaf_makes_the_iterations_pi() {
        for (of, length) in [(1usize, 150usize), (2, 70)] {
            let e = series(of, length, 11 + of as u64);
            let start = || Degrees {
                degrees: (0..2 * WIDTH * of).map(|c| 1 + c % 3).collect(),
                quiet: vec![0; 2 * WIDTH * of],
            };
            let mut whole = start();
            let reference = one_at_a_time(&e, &mut whole);
            for leaf in [2, 3, 7, 16, 64] {
                let mut state = start();
                let pi = steps(&e, &mut state, leaf);
                assert_eq!(pi, reference, "{of} words, leaf {leaf}: π");
                assert_eq!(state.degrees, whole.degrees, "{of} words, leaf {leaf}: δ");
                assert_eq!(state.quiet, whole.quiet, "{of} words, leaf {leaf}: quiet");
            }
        }
    }

    /// `E·π` vanishes below `X^k`, which is what makes the second half's
    /// `E` the quotient; and each column of `π` is no longer than its `δ`
    /// grew, so that `F·π` keeps the degrees the iteration claims.
    #[test]
    fn e_times_pi_vanishes_below_its_length() {
        for of in [1usize, 2] {
            let length = 90;
            let e = series(of, length, 5 + of as u64);
            let count = 2 * WIDTH * of;
            let mut state = Degrees {
                degrees: vec![1; count],
                quiet: vec![0; count],
            };
            let pi = steps(&e, &mut state, 8);
            let made = product(&e, &pi);
            for index in 0..length {
                for c in 0..count {
                    assert!(
                        made.column(index, c).iter().all(|&word| word == 0),
                        "{of} words: coefficient {index}, column {c}"
                    );
                }
            }
            for c in 0..count {
                let grew = state.degrees[c] - 1;
                for index in grew + 1..pi.length() {
                    assert!(
                        pi.column(index, c).iter().all(|&word| word == 0),
                        "{of} words: column {c} past {grew}"
                    );
                }
            }
        }
    }
}
