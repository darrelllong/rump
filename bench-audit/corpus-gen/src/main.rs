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

    let mut index = String::new();
    let _ = writeln!(index, "# corpus index, generator-version {GENERATOR_VERSION}");
    for c in &corpora {
        c.write(&dir);
        let _ = writeln!(index, "{}", c.name);
    }
    fs::write(dir.join("INDEX.txt"), index).expect("write index");
    eprintln!("wrote {} corpora to {}", corpora.len(), dir.display());
}
