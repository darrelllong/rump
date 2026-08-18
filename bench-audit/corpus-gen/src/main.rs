//! Deterministic corpus generator for the Rump public-API performance audit.
//!
//! Writes named corpora as text so that both revision adapters provably parse
//! the *same bytes*. The alternative — generating operands inside each adapter
//! from a shared seed — would depend on two copies of a generator staying
//! identical, and a divergence there is invisible: it does not fail, it just
//! gives the two revisions different work. Serialized input removes that class
//! of error entirely.
//!
//! Depends on nothing, so the corpus cannot vary with a Rump revision.
//!
//! Format: one operand per line, lowercase hex, no prefix, most significant
//! digit first, `-` prefix for a negative integer. A `#` line is a comment
//! recording the case parameters. Blank lines separate operand groups.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

/// The generator's own version, recorded in the manifest. Bump it whenever the
/// emitted bytes change for a fixed seed, so a stale corpus cannot be compared
/// against a fresh one without the difference being visible.
const GENERATOR_VERSION: &str = "1";

/// SplitMix64. Fixed here so a corpus is reproducible from its seed forever,
/// independent of any library.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Operand classes the audit requires. Each is a distinct dispatch and
/// carry-behaviour population, not a cosmetic variation.
#[derive(Clone, Copy, Debug)]
enum Class {
    /// Uniformly scattered limbs.
    Dense,
    /// Mostly zero limbs: exercises normalization and early exits.
    Sparse,
    /// All ones: maximal carry and borrow propagation.
    AllOnes,
    /// `2^k - 1`: one below a power of two.
    BelowPowerOfTwo,
    /// `2^k + 1`: one above a power of two.
    AbovePowerOfTwo,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Dense => "dense",
            Class::Sparse => "sparse",
            Class::AllOnes => "allones",
            Class::BelowPowerOfTwo => "below2k",
            Class::AbovePowerOfTwo => "above2k",
        }
    }
}

/// A big-endian hex integer of `bits` bits in the given class.
fn operand(bits: usize, class: Class, rng: &mut Rng) -> String {
    let limbs = bits.div_ceil(64);
    let mut words = vec![0u64; limbs];
    match class {
        Class::Dense => {
            for w in words.iter_mut() {
                *w = rng.next();
            }
        }
        Class::Sparse => {
            // About one set limb in eight, and a set low bit so the value is
            // odd where the caller needs it to be.
            for w in words.iter_mut() {
                *w = if rng.next() % 8 == 0 { rng.next() } else { 0 };
            }
        }
        Class::AllOnes => {
            for w in words.iter_mut() {
                *w = u64::MAX;
            }
        }
        Class::BelowPowerOfTwo | Class::AbovePowerOfTwo => {
            // 2^(bits-1) then adjust, so the width is exactly `bits`.
            for w in words.iter_mut() {
                *w = 0;
            }
            let top = (bits - 1) / 64;
            words[top] |= 1u64 << ((bits - 1) % 64);
            if matches!(class, Class::BelowPowerOfTwo) {
                // 2^(bits-1) - 1 + 2^(bits-1) = 2^bits - 1 would widen; use
                // 2^(bits-1) - 1 instead, which is `bits-1` wide and all ones.
                for w in words.iter_mut() {
                    *w = 0;
                }
                for i in 0..top {
                    words[i] = u64::MAX;
                }
                let rem = (bits - 1) % 64;
                if rem > 0 {
                    words[top] = (1u64 << rem) - 1;
                }
            } else {
                words[0] |= 1;
            }
        }
    }
    // Force the declared width for the random classes.
    if matches!(class, Class::Dense | Class::Sparse) {
        let top = (bits - 1) / 64;
        for w in words.iter_mut().skip(top + 1) {
            *w = 0;
        }
        words[top] |= 1u64 << ((bits - 1) % 64);
    }
    let mut hex = String::new();
    let mut leading = true;
    for w in words.iter().rev() {
        if leading {
            let _ = write!(hex, "{w:x}");
            leading = false;
        } else {
            let _ = write!(hex, "{w:016x}");
        }
    }
    if hex.is_empty() {
        hex.push('0');
    }
    hex
}

/// An odd operand, for moduli that require it.
fn odd_operand(bits: usize, class: Class, rng: &mut Rng) -> String {
    let mut hex = operand(bits, class, rng);
    // Set the low bit by fixing the last hex digit to an odd value.
    if let Some(last) = hex.pop() {
        let digit = u8::from_str_radix(&last.to_string(), 16).unwrap_or(1) | 1;
        let _ = write!(hex, "{digit:x}");
    }
    hex
}

/// GF(2^m) field polynomials and elements.
///
/// The field polynomial is a named constant, not random: an irreducible
/// polynomial cannot be drawn by chance. Elements are emitted as distinct
/// classes because the field's fast paths differ on them.
fn gf2m_corpus(degree: usize, poly_hex: &str, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("gf2m-{degree}-{count}"));
    c.note("kind", "gf2m")
        .note("degree", &degree.to_string())
        .note("field-polynomial", poly_hex)
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "line 1 is the field polynomial; then pairs of elements");
    c.lines.push(poly_hex.to_string());
    // Distinct element classes: zero, one, sparse, dense.
    c.lines.push("0".to_string());
    c.lines.push("1".to_string());
    for _ in 0..count {
        c.lines.push(operand(degree - 1, Class::Dense, &mut rng));
        c.lines.push(operand(degree - 1, Class::Sparse, &mut rng));
    }
    c
}

/// Lattice bases: `dimension` rows of `dimension` signed entries.
fn lattice_corpus(dimension: usize, bits: usize, shape: &str, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("lattice-{dimension}-{bits}-{shape}-{count}"));
    c.note("kind", "lattice")
        .note("dimension", &dimension.to_string())
        .note("bits", &bits.to_string())
        .note("shape", shape)
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "dimension^2 entries per basis, row-major; blank line ends a basis");
    for _ in 0..count {
        for row in 0..dimension {
            for col in 0..dimension {
                let entry = match shape {
                    // Full rank: random off-diagonal, a large diagonal.
                    "random" => {
                        if row == col {
                            operand(bits, Class::Dense, &mut rng)
                        } else {
                            operand(bits / 2, Class::Dense, &mut rng)
                        }
                    }
                    // Ill-conditioned: one huge row against small ones.
                    "illcond" => {
                        if row == 0 {
                            operand(bits, Class::Dense, &mut rng)
                        } else {
                            operand(bits / 8 + 1, Class::Dense, &mut rng)
                        }
                    }
                    // Nearly dependent: rows differ by a small perturbation.
                    "neardep" => {
                        if col == 0 {
                            operand(bits, Class::Dense, &mut rng)
                        } else if row == col {
                            "1".to_string()
                        } else {
                            "0".to_string()
                        }
                    }
                    _ => panic!("unknown lattice shape {shape}"),
                };
                c.lines.push(entry);
            }
        }
        c.lines.push(String::new());
    }
    c
}

/// A batch of same-width values, for the tree and batch-inverse families.
fn batch_corpus(size: usize, bits: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("batch-{size}-{bits}"));
    c.note("kind", "singles")
        .note("batch-size", &size.to_string())
        .note("bits", &bits.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..size {
        c.lines.push(odd_operand(bits, Class::Dense, &mut rng));
    }
    c
}

/// Sampling bounds: immediately below a power of two (favourable) and
/// immediately above half a power of two (rejection-heavy).
fn bound_corpus(bits: usize, shape: &str, count: usize) -> Corpus {
    let mut c = Corpus::new(&format!("bound-{bits}-{shape}-{count}"));
    c.note("kind", "singles")
        .note("bits", &bits.to_string())
        .note("shape", shape)
        .note("count", &count.to_string());
    // Both bounds are stated exactly, not drawn: the point of the pair is the
    // rejection rate, which is a property of where the bound sits relative to
    // the nearest power of two, so a random draw would defeat it.
    //
    //   below2k    2^bits - 1      every draw under 2^bits is accepted
    //   abovehalf  2^(bits-1) + 1  just over half, so nearly half are rejected
    let value = match shape {
        "below2k" => "f".repeat(bits / 4),
        "abovehalf" => {
            let mut v = String::from("8");
            v.push_str(&"0".repeat(bits / 4 - 2));
            v.push('1');
            v
        }
        _ => panic!("unknown bound shape {shape}"),
    };
    for _ in 0..count {
        c.lines.push(value.clone());
    }
    c
}

/// Primes for the modular-square-root workload, at two widths of congruence
/// class each.
///
/// `mod_sqrt` requires a prime modulus. A random odd integer is essentially
/// never prime, so a corpus of random moduli measures only the rejection path
/// and returns no root at all — which is what the first version of this audit
/// did. These are stated as constants because a prime cannot be drawn by
/// chance at these widths.
///
/// The class matters as much as the width: for `p ≡ 3 (mod 4)` a root is one
/// exponentiation, `a^((p+1)/4)`, while `p ≡ 1 (mod 4)` needs Tonelli–Shanks.
/// Benchmarking only the first class would miss the algorithm entirely.
///
/// Generated with a fixed seed and confirmed by 64 Miller–Rabin rounds; each
/// is verifiably prime and verifiably in its stated class.
const PRIMES: &[(usize, u32, &str)] = &[
    // 256-bit prime, p ≡ 1 (mod 4)
    (256, 1, "8f34d4c56961cb5d44478eac87ce893da4926c7d76f3b12e61b3f2747b6f23b9"),
    // 256-bit prime, p ≡ 3 (mod 4)
    (256, 3, "aecd810e4e97d19e4e9b0915a67537adbc5d47de6ed4593020462b476f3e4163"),
    // 512-bit prime, p ≡ 1 (mod 4)
    (512, 1, "df6e5bac7ef1ccc7496bac47aa4a07878198fe9a3116c1cab67de3a765d66754450cd60be1e091267d6d1cb826cac1104966d0717a5a5a4f4279877b916f757d"),
    // 512-bit prime, p ≡ 3 (mod 4)
    (512, 3, "bae18736560674b25ad56057e7aa49d715b888320aed8d386a9ee7a97888619a1114544629f841d924a4cb5b832bb10585d8d8bf8a0f75f803ad66cfd14e77d7"),
    // 1024-bit prime, p ≡ 1 (mod 4)
    (1024, 1, "8b8a722f023d5f078bb3042fbf4c813d95153ee32e8b46fcb9921dc9a69fa7c71587986955a24dc596b8d59dcaa6d564cf127d50d5e2dc3f82d0014bf614edb9502ba14af80235a0e4ac22c3d57001c763664273ceb944969e3fbdf5aa5b84f4a756ff06e268acf644278480ecea45530e17e87cbae8c5cc9ea6b9f20215ffc9"),
    // 1024-bit prime, p ≡ 3 (mod 4)
    (1024, 3, "ffe5bbfabfc587b6a5cf1c6bdc2497767d87eebe860465d3bf5dff42953d336d69ca0856bdd4e5330109440ce0f39e421a981df273b37e13ec5559692d3b089ca0cc37129cf8209a1737c28b51b85f340c325735970444a3532aba4396025b5db81646e7a433521d405c770ae6c25137b312c2e8a218c8c7898e7d46d3cc2bb3"),
    // 2048-bit prime, p ≡ 1 (mod 4)
    (2048, 1, "b5ec3589cf4d219c9241ef1157ff7635c818342c672a030093030cbb8bb21158da7bc5f2756df931d3823ac469fc53acbdb6ae9b760dfbf5176257073d65053f728bfd48e225306d172973abd7112f4aed522573f399e505fa0c885262b29d5b2bcae1eaa7b6f4573757b792ef2a30d818653ef70d540a3e1a343809fe775ab07256eee692be56aa28ed8ba670d9afe486c176b4864c5ff1ad04d84ec80e9d8c6edfe3400dad3e6d64f38e82bd6e283eafeddaeac08a86db29c2a0be4406e9d641db31af1895d8ce0485fb19ddc5868eff6dfb119a451b375830c73987537affcc98b39085930184c637ae0f7e5c8c8ea337e8ae4c2f424301e18ce1ae750b39"),
    // 2048-bit prime, p ≡ 3 (mod 4)
    (2048, 3, "bdcab2268542ff510e3d8ebf2f8ebd1b1651a41787d5f745d8bc3eefabf3f05708defc1d083cf6f1daa0b1c734c5c23b098330d30dee5147f4e2de4b8b7352aff0109b577f45952a50322046bf434d620f60ec5774539f260791ffbb5abcbfc3df3b71a54c971bf68b0f28647c7010f608cf945b41d545a582963c3baf006b594bd988dc375eb1d5837114a523593a0e775e57bc28a74c7df4fac668df5bb974c003bbaa55aec4bd0f2e225788df81bb3ea6e7ee1e071dbddd4e9fb2de8a21b16b5cba5e9e0dd34a60054401067945660b064a44b27a39305609420f80b15e4bcfd4ce26299217b7b6ed175a01b1efe84fe45d2619b141f8575af142248c2447"),
];

/// Values to take square roots of, modulo a stated prime. About half are
/// quadratic residues, so about half the calls return a root and half take
/// the non-residue exit — both paths are on the measured path, in their
/// natural proportion.
fn prime_corpus(bits: usize, class: u32, poly: &str, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("prime-{bits}-{class}mod4-{count}"));
    c.note("kind", "prime-modulus")
        .note("bits", &bits.to_string())
        .note("class", &format!("{class} mod 4"))
        .note("modulus", poly)
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "line 1 is the prime modulus; the rest are values");
    c.lines.push(poly.to_string());
    for _ in 0..count {
        c.lines.push(operand(bits - 1, Class::Dense, &mut rng));
    }
    c
}

/// A GF(2) relation matrix in the shape the sieve produces: many more columns
/// than the handful of nonzeros in any one row.
///
/// Rows are packed little-endian by column — column `c` lives at bit `c % 64`
/// of word `c / 64` — which is the packing `rump::gf2` documents. The row is
/// emitted as space-separated hex words so the adapter can read it back
/// without a bit-level parser.
///
/// `columns + excess` rows: the null space of a matrix over GF(2) with more
/// rows than columns has dimension at least the excess, so the solvers are
/// guaranteed something to find.
fn gf2_matrix_corpus(columns: usize, excess: usize, weight: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let rows = columns + excess;
    let words = columns.div_ceil(64);
    let mut c = Corpus::new(&format!("gf2mat-{columns}-{excess}-{weight}"));
    c.note("kind", "gf2-matrix")
        .note("columns", &columns.to_string())
        .note("rows", &rows.to_string())
        .note("row-weight", &weight.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "one row per line, space-separated hex words, little-endian by column");
    for _ in 0..rows {
        let mut row = vec![0u64; words];
        for _ in 0..weight {
            let bit = (rng.next() as usize) % columns;
            row[bit / 64] ^= 1u64 << (bit % 64);
        }
        let text: Vec<String> = row.iter().map(|w| format!("{w:x}")).collect();
        c.lines.push(text.join(" "));
    }
    c
}

/// Integer polynomials for the real-root workload.
///
/// `split` builds `∏ (x − r_i)` over distinct small integers, so the root
/// count is known exactly and the isolator has the maximum amount of work to
/// do. `generic` uses random coefficients, where most roots are complex and
/// the isolator exits early. The two bracket the cost.
fn poly_roots_corpus(degree: usize, shape: &str, bits: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("poly-{degree}-{shape}-{bits}"));
    c.note("kind", "polynomial")
        .note("degree", &degree.to_string())
        .note("shape", shape)
        .note("bits", &bits.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "one signed decimal coefficient per line, constant term first");
    let coefficients: Vec<i128> = match shape {
        "split" => {
            // ∏ (x − r) for r = 1..=degree: `degree` distinct real roots.
            //
            // The degree is capped low, for two separate reasons.
            //
            // The constant term is `degree!`, which passes i128 at degree 34.
            // The arithmetic below is checked rather than wrapping, because a
            // release build wraps silently: an earlier version of this
            // generator emitted a degree-64 "split" polynomial whose
            // coefficients were wrap artefacts and whose roots were not
            // 1..64 at all.
            //
            // The binding limit is smaller and is conditioning, not range.
            // `∏ (x − r)` over `r = 1..n` is Wilkinson's polynomial, whose
            // roots are famously hypersensitive to its coefficients; since
            // `real_roots` isolates over f64, the isolator is working on a
            // different polynomial than the one written down. Measured: at
            // degree 32 it returns 8 roots rather than 32, drifting to 8.07
            // where 8 was meant, and at degree 16 the middle roots are
            // already wrong in the fifth decimal. Neither is a defect in
            // `real_roots` — it is what f64 root isolation does to an
            // ill-conditioned basis — but both make a poor benchmark input
            // and a digest that would not survive a change of host rounding.
            // 12 is the largest degree at which every root returns exact.
            assert!(
                degree <= 12,
                "split degree {degree} is past the point where f64 isolation \
                 recovers the roots; see the note above"
            );
            let mut poly: Vec<i128> = vec![1];
            for r in 1..=degree as i128 {
                let mut next = vec![0i128; poly.len() + 1];
                for (i, &a) in poly.iter().enumerate() {
                    next[i + 1] = next[i + 1].checked_add(a).expect("coefficient fits i128");
                    next[i] = next[i]
                        .checked_sub(a.checked_mul(r).expect("coefficient fits i128"))
                        .expect("coefficient fits i128");
                }
                poly = next;
            }
            poly
        }
        "generic" => (0..=degree)
            .map(|_| {
                let magnitude = (rng.next() % (1u64 << bits.min(62))) as i128;
                if rng.next() & 1 == 0 { magnitude } else { -magnitude }
            })
            .collect(),
        _ => panic!("unknown polynomial shape {shape}"),
    };
    for a in coefficients {
        c.lines.push(a.to_string());
    }
    c
}

/// A streaming workload, declared rather than enumerated.
///
/// The point of these cells is a working set past the last-level cache, which
/// at 96 MiB would mean a 192 MiB hex corpus — the file would cost more to
/// move between hosts than the measurement costs to take. So the corpus
/// carries only the header, and both adapters expand it from the stated seed
/// with the same generator. Determinism is unaffected: the expansion lives in
/// the byte-identical `shared.rs`, and the digest check confirms the two
/// revisions built the same operands.
///
/// Working set per traversal is `count × 3 × bits / 8` bytes: two operands
/// read and one result written.
fn stream_corpus(bits: usize, count: usize, seed: u64) -> Corpus {
    let bytes = count * 3 * bits / 8;
    let mut c = Corpus::new(&format!("stream-{bits}-{count}"));
    c.note("kind", "streaming")
        .note("bits", &bits.to_string())
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("working-set-bytes", &bytes.to_string())
        .note("layout", "header only; the adapter expands the operands from the seed");
    c
}

struct Corpus {
    name: String,
    header: Vec<String>,
    lines: Vec<String>,
}

impl Corpus {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            header: Vec::new(),
            lines: Vec::new(),
        }
    }
    fn note(&mut self, key: &str, value: &str) -> &mut Self {
        self.header.push(format!("# {key}: {value}"));
        self
    }
    fn write(&self, dir: &PathBuf) {
        let mut text = String::new();
        let _ = writeln!(text, "# corpus: {}", self.name);
        let _ = writeln!(text, "# generator-version: {GENERATOR_VERSION}");
        for h in &self.header {
            let _ = writeln!(text, "{h}");
        }
        for l in &self.lines {
            let _ = writeln!(text, "{l}");
        }
        fs::write(dir.join(format!("{}.txt", self.name)), text).expect("write corpus");
    }
}

/// Pairs of same-width operands: the input for add, sub, mul, cmp, gcd.
fn pair_corpus(bits: usize, class: Class, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("pair-{bits}-{}-{count}", class.name()));
    c.note("kind", "pairs")
        .note("bits", &bits.to_string())
        .note("class", class.name())
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..count {
        c.lines.push(operand(bits, class, &mut rng));
        c.lines.push(operand(bits, class, &mut rng));
    }
    c
}

/// Pairs with a declared width ratio, for the unbalanced-product paths.
fn ratio_corpus(long_bits: usize, short_bits: usize, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("ratio-{long_bits}x{short_bits}-{count}"));
    c.note("kind", "pairs")
        .note("long-bits", &long_bits.to_string())
        .note("short-bits", &short_bits.to_string())
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..count {
        c.lines.push(operand(long_bits, Class::Dense, &mut rng));
        c.lines.push(operand(short_bits, Class::Dense, &mut rng));
    }
    c
}

/// Dividend/divisor pairs of a named shape.
fn division_corpus(bits: usize, shape: &str, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("div-{bits}-{shape}-{count}"));
    c.note("kind", "pairs")
        .note("bits", &bits.to_string())
        .note("shape", shape)
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..count {
        let dividend = operand(bits, Class::Dense, &mut rng);
        let divisor = match shape {
            "full" => operand(bits, Class::Dense, &mut rng),
            "half" => operand(bits / 2, Class::Dense, &mut rng),
            "word" => operand(64, Class::Dense, &mut rng),
            // A divisor just under the dividend: the one-digit-quotient case.
            "near" => dividend.clone(),
            _ => panic!("unknown division shape {shape}"),
        };
        c.lines.push(dividend);
        c.lines.push(divisor);
    }
    c
}

/// Odd moduli, for Montgomery and the modular families.
fn modulus_corpus(bits: usize, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("modulus-odd-{bits}-{count}"));
    c.note("kind", "singles")
        .note("bits", &bits.to_string())
        .note("parity", "odd")
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..count {
        c.lines.push(odd_operand(bits, Class::Dense, &mut rng));
    }
    c
}

/// Base/exponent/modulus triples for exponentiation.
fn exponent_corpus(modulus_bits: usize, exponent: &str, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("exp-{modulus_bits}-{exponent}-{count}"));
    c.note("kind", "triples")
        .note("modulus-bits", &modulus_bits.to_string())
        .note("exponent", exponent)
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"));
    for _ in 0..count {
        let modulus = odd_operand(modulus_bits, Class::Dense, &mut rng);
        let base = operand(modulus_bits, Class::Dense, &mut rng);
        let e = match exponent {
            "zero" => "0".to_string(),
            "one" => "1".to_string(),
            "f4" => "10001".to_string(),
            "e256" => operand(256, Class::Dense, &mut rng),
            "full" => operand(modulus_bits, Class::Dense, &mut rng),
            _ => panic!("unknown exponent class {exponent}"),
        };
        c.lines.push(base);
        c.lines.push(e);
        c.lines.push(modulus);
    }
    c
}

/// Polynomial coefficient lists: degree+1 coefficients per polynomial, two
/// polynomials per case.
fn poly_corpus(degree: usize, coefficient_bits: usize, count: usize, seed: u64) -> Corpus {
    let mut rng = Rng(seed);
    let mut c = Corpus::new(&format!("poly-{degree}-{coefficient_bits}-{count}"));
    c.note("kind", "polynomials")
        .note("degree", &degree.to_string())
        .note("coefficient-bits", &coefficient_bits.to_string())
        .note("count", &count.to_string())
        .note("seed", &format!("{seed:#x}"))
        .note("layout", "one line per coefficient, ascending; blank line ends a polynomial");
    for _ in 0..count {
        for _ in 0..2 {
            for _ in 0..=degree {
                c.lines.push(operand(coefficient_bits, Class::Dense, &mut rng));
            }
            c.lines.push(String::new());
        }
    }
    c
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: corpus-gen <output-dir>");
    fs::create_dir_all(&dir).expect("create corpus dir");

    let mut corpora: Vec<Corpus> = Vec::new();

    // Integers: the ordinary widths, plus the dispatch neighbourhoods that
    // matter for multiplication (Karatsuba at 32 limbs = 2048 bits, Toom-3 at
    // 128 limbs = 8192 bits).
    for &bits in &[64usize, 256, 1024, 4096, 16384] {
        for class in [Class::Dense, Class::Sparse, Class::AllOnes] {
            corpora.push(pair_corpus(bits, class, 32, 0x5eed_0000 ^ bits as u64));
        }
    }
    for &bits in &[1984usize, 2048, 2112, 8128, 8192, 8256] {
        corpora.push(pair_corpus(bits, Class::Dense, 32, 0xd150_0000u64 ^ bits as u64));
    }
    for &bits in &[256usize, 4096] {
        corpora.push(pair_corpus(bits, Class::BelowPowerOfTwo, 32, 0xb0b0 ^ bits as u64));
        corpora.push(pair_corpus(bits, Class::AbovePowerOfTwo, 32, 0xa1a1 ^ bits as u64));
    }
    // Width ratios near 3:2, 2:1 and 8:1.
    corpora.push(ratio_corpus(3072, 2048, 32, 0x3a20));
    corpora.push(ratio_corpus(4096, 2048, 32, 0x2a10));
    corpora.push(ratio_corpus(16384, 2048, 32, 0x8a10));
    // Division shapes.
    for &bits in &[1024usize, 4096] {
        for shape in ["full", "half", "word", "near"] {
            corpora.push(division_corpus(bits, shape, 32, 0xd140 ^ bits as u64));
        }
    }
    // Moduli and exponents.
    for &bits in &[64usize, 256, 1024, 2048, 4096] {
        corpora.push(modulus_corpus(bits, 16, 0x9d0d ^ bits as u64));
    }
    for &bits in &[256usize, 2048] {
        for e in ["zero", "one", "f4", "e256", "full"] {
            corpora.push(exponent_corpus(bits, e, 8, 0xe0e0 ^ bits as u64));
        }
    }
    // Polynomials at and around the convolution cutoffs.
    for &(degree, cbits) in &[(8usize, 64usize), (32, 64), (96, 64), (128, 64), (32, 256)] {
        corpora.push(poly_corpus(degree, cbits, 8, 0xf01d ^ degree as u64));
    }

    // GF(2^m): the two fields the crate already tests, plus a smaller generic
    // one. The polynomials are the standard reduction trinomials/pentanomials.
    // x^163 + x^7 + x^6 + x^3 + 1
    corpora.push(gf2m_corpus(163, "800000000000000000000000000000000000000c9", 8, 0x9163));
    // x^233 + x^74 + 1
    corpora.push(gf2m_corpus(
        233,
        "20000000000000000000000000000000000000000000000000000400001",
        8,
        0x9233,
    ));
    // x^571 + x^10 + x^5 + x^2 + 1
    corpora.push(gf2m_corpus(
        571,
        "80000000000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000425",
        8,
        0x9571,
    ));

    // Lattices.
    for &dim in &[2usize, 8, 16, 32] {
        for &bits in &[32usize, 128, 512] {
            corpora.push(lattice_corpus(dim, bits, "random", 4, 0x1a77 ^ (dim * bits) as u64));
        }
    }
    corpora.push(lattice_corpus(8, 128, "illcond", 4, 0x111c));
    corpora.push(lattice_corpus(8, 128, "neardep", 4, 0x11de));

    // Batch algorithms.
    for &size in &[1usize, 8, 64, 512, 4096] {
        for &bits in &[64usize, 256, 1024] {
            corpora.push(batch_corpus(size, bits, 0xba7c ^ (size * bits) as u64));
        }
    }

    // Modular square roots, over genuinely prime moduli.
    for &(bits, class, hex) in PRIMES {
        corpora.push(prime_corpus(bits, class, hex, 32, 0x9047 ^ (bits * class as usize) as u64));
    }

    // GF(2) relation matrices, at the shape and weight a sieve produces.
    for &(columns, excess) in &[(512usize, 16usize), (2048, 32), (8192, 64)] {
        for &weight in &[8usize, 24] {
            corpora.push(gf2_matrix_corpus(columns, excess, weight, 0x9f2 ^ (columns * weight) as u64));
        }
    }

    // Integer polynomials for real-root isolation.
    for &degree in &[4usize, 8, 12] {
        corpora.push(poly_roots_corpus(degree, "split", 32, 0x9013 ^ degree as u64));
    }
    for &degree in &[4usize, 16, 64] {
        corpora.push(poly_roots_corpus(degree, "generic", 32, 0x9014 ^ degree as u64));
    }

    // Streaming: one working set well past any last-level cache, and one
    // that fits comfortably inside it, so the pair brackets the crossover.
    corpora.push(stream_corpus(524_288, 512, 0x57ea_a001));
    corpora.push(stream_corpus(524_288, 16, 0x57ea_a002));

    // Sampling bounds.
    for &bits in &[256usize, 2048] {
        corpora.push(bound_corpus(bits, "below2k", 16));
        corpora.push(bound_corpus(bits, "abovehalf", 16));
    }

    let mut index = String::new();
    let _ = writeln!(index, "# corpus index, generator-version {GENERATOR_VERSION}");
    for c in &corpora {
        c.write(&dir);
        let _ = writeln!(index, "{}", c.name);
    }
    fs::write(dir.join("INDEX.txt"), index).expect("write index");
    eprintln!("wrote {} corpora to {}", corpora.len(), dir.display());
}
