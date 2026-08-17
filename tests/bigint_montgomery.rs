//! Differential and invariant tests for Montgomery arithmetic.
//!
//! Every exponentiation is checked against a naive square-and-multiply ladder
//! built on the division-based [`BigUint::mod_mul`] — a reference that shares
//! no code with the Montgomery kernels (separate multiply, separate reduction,
//! no Montgomery domain), so agreement is evidence rather than a restatement.
//!
//! The input families deliberately straddle the implementation's seams: the
//! 64-bit boundary where `pow` switches from binary to windowed scanning, the
//! 4-bit window alignments, all-zero windows (whose multiply is skipped),
//! and operand shapes that stress the reduction's conditional subtract.

use rump::modular::ModulusError;
use rump::modular::MontgomeryContext;
use rump::BigUint;

/// Deterministic test generator: splitmix64 (Steele, Lea & Flood 2014),
/// vendored so the tests need no dependency. Not a CSPRNG and not meant to
/// be one — the tests only need reproducible, well-scattered operand draws.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let word = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

/// Independent reference: right-to-left square-and-multiply over the
/// division-based `mod_mul`.
fn pow_reference(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
    if modulus.is_one() {
        return BigUint::zero();
    }

    let mut result = BigUint::one();
    let mut power = base.rem(modulus);
    for bit in 0..exponent.bits() {
        if exponent.bit(bit) {
            result = BigUint::mod_mul(&result, &power, modulus);
        }
        power = BigUint::mod_mul(&power, &power, modulus);
    }
    result
}

fn from_limbs(limbs: &[u64]) -> BigUint {
    let mut bytes = Vec::with_capacity(limbs.len() * 8);
    for &limb in limbs.iter().rev() {
        bytes.extend_from_slice(&limb.to_be_bytes());
    }
    BigUint::from_be_bytes(&bytes)
}

fn rng() -> SplitMix64 {
    SplitMix64::new(0xa7a7_a7a7_a7a7_a7a7)
}

fn random_limb(rng: &mut SplitMix64) -> u64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    u64::from_le_bytes(bytes)
}

/// Limbs biased toward carry/borrow edge values; uniform draws essentially
/// never produce the all-ones and power-of-two limbs where reduction and
/// conditional-subtract mistakes live.
fn structured_limb(rng: &mut SplitMix64) -> u64 {
    let raw = random_limb(rng);
    match raw % 8 {
        0 => 0,
        1 => u64::MAX,
        2 => 1,
        3 => 1 << 63,
        4 => (1 << 63) | 1,
        5 => u64::MAX - 1,
        _ => random_limb(rng),
    }
}

fn structured_biguint(words: usize, rng: &mut SplitMix64) -> BigUint {
    let limbs: Vec<u64> = (0..words).map(|_| structured_limb(rng)).collect();
    from_limbs(&limbs)
}

fn structured_odd_modulus(words: usize, rng: &mut SplitMix64) -> BigUint {
    let mut limbs: Vec<u64> = (0..words).map(|_| structured_limb(rng)).collect();
    limbs[0] |= 1;
    if limbs[words - 1] == 0 {
        limbs[words - 1] = structured_limb(rng) | 1;
    }
    from_limbs(&limbs)
}

/// Check `ctx.pow` (and `pow_residue` through the same call) against the
/// reference, plus the canonical-range invariant.
fn check_pow(ctx: &MontgomeryContext, base: &BigUint, exponent: &BigUint) {
    let modulus = ctx.modulus();
    let actual = ctx.pow(base, exponent);
    let expected = pow_reference(base, exponent, modulus);
    assert_eq!(
        actual, expected,
        "pow disagrees with reference\n  base = {base:?}\n  exp = {exponent:?}\n  n = {modulus:?}"
    );
    assert!(actual < *modulus || modulus.is_one());

    let encoded = ctx.to_residue(&base.rem(modulus));
    assert_eq!(
        ctx.from_residue(&ctx.pow_residue(&encoded, exponent).expect("same context"))
            .expect("same context"),
        expected,
        "pow_residue disagrees with pow for base {base:?}"
    );
}

#[test]
fn pow_matches_reference_across_shapes() {
    let mut rng = rng();

    for modulus_words in [1usize, 2, 3, 4, 8] {
        for _ in 0..6 {
            let modulus = structured_odd_modulus(modulus_words, &mut rng);
            let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

            for base_words in [1usize, modulus_words, modulus_words + 1] {
                let base = structured_biguint(base_words, &mut rng);
                for exponent_words in [1usize, 2, 4] {
                    let exponent = structured_biguint(exponent_words, &mut rng);
                    check_pow(&ctx, &base, &exponent);
                }
            }
        }
    }
}

#[test]
fn pow_exponent_boundaries() {
    // The path switch sits at 64 exponent bits and the window scan works in
    // 4-bit strides: pin exponents on, either side of, and straddling both
    // seams, plus runs of zero windows (their multiply is skipped) and
    // all-ones runs (every window multiplies).
    let mut rng = rng();

    let mut exponents = vec![
        BigUint::zero(),
        BigUint::one(),
        BigUint::from_u64(2),
        BigUint::from_u64(3),
        BigUint::from_u64(65_537),
        BigUint::from_u64(u64::MAX - 1),
        BigUint::from_u64(u64::MAX),
        BigUint::from_u128(1u128 << 64),
        BigUint::from_u128((1u128 << 64) | 1),
        BigUint::from_u128(u128::MAX),
    ];
    // 2^k and 2^k - 1 around the window strides: bit lengths 63..=66 and
    // 127..=129 cover bits % 4 of every phase on both sides of the switch.
    for bit in [63usize, 64, 65, 66, 127, 128, 129] {
        let mut power_of_two = BigUint::zero();
        power_of_two.set_bit(bit);
        exponents.push(power_of_two.sub(&BigUint::one()));
        exponents.push(power_of_two);
    }
    // A long exponent whose middle is a full 64-bit run of zeros: sixteen
    // consecutive zero windows exercise the skipped-multiply path.
    exponents.push(from_limbs(&[0x8000_0000_0000_0001, 0, u64::MAX]));

    for modulus_words in [1usize, 2, 4] {
        let modulus = structured_odd_modulus(modulus_words, &mut rng);
        let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");
        let base = structured_biguint(modulus_words, &mut rng);
        for exponent in &exponents {
            check_pow(&ctx, &base, exponent);
        }
    }
}

#[test]
fn pow_base_edge_cases() {
    let mut rng = rng();
    let modulus = structured_odd_modulus(4, &mut rng);
    let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

    let n_minus_1 = modulus.sub(&BigUint::one());
    let n_plus_1 = modulus.add(&BigUint::one());
    let twice_n = modulus.add(&modulus);
    let bases = [
        BigUint::zero(),
        BigUint::one(),
        BigUint::from_u64(2),
        n_minus_1,
        modulus.clone(), // congruent to zero
        n_plus_1,        // congruent to one
        twice_n.add(&BigUint::from_u64(3)),
    ];
    let exponents = [
        BigUint::zero(),
        BigUint::one(),
        BigUint::from_u64(65_537),
        structured_biguint(3, &mut rng),
    ];

    for base in &bases {
        for exponent in &exponents {
            check_pow(&ctx, base, exponent);
        }
    }
}

#[test]
fn pow_tiny_and_degenerate_moduli() {
    // modulus == 1: everything collapses to zero.
    let one_ctx = MontgomeryContext::new(&BigUint::one()).expect("one is odd");
    assert_eq!(
        one_ctx.pow(&BigUint::from_u64(5), &BigUint::from_u64(3)),
        BigUint::zero()
    );
    assert_eq!(
        one_ctx.pow(&BigUint::from_u64(5), &BigUint::zero()),
        BigUint::zero()
    );

    // Small odd moduli, exhaustively checkable shapes.
    for n in [3u64, 5, 9, 15, 255, 65_535] {
        let modulus = BigUint::from_u64(n);
        let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");
        for base in 0..7u64 {
            for exponent in 0..7u64 {
                check_pow(&ctx, &BigUint::from_u64(base), &BigUint::from_u64(exponent));
            }
        }
    }

    // Even or zero moduli have no Montgomery context.
    assert_eq!(
        MontgomeryContext::new(&BigUint::zero()),
        Err(ModulusError::Zero)
    );
    assert_eq!(
        MontgomeryContext::new(&BigUint::from_u64(100)),
        Err(ModulusError::Even)
    );
}

#[test]
fn encode_decode_and_context_constants() {
    let mut rng = rng();
    for modulus_words in [1usize, 2, 4, 8] {
        for _ in 0..8 {
            let modulus = structured_odd_modulus(modulus_words, &mut rng);
            if modulus.is_one() {
                continue;
            }
            let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

            // one_mont really is the encoding of one, and decode inverts
            // encode across the whole residue range shape.
            assert_eq!(ctx.to_residue(&BigUint::one()), ctx.one());
            assert_eq!(
                ctx.from_residue(&ctx.one()).expect("same context"),
                BigUint::one()
            );

            for _ in 0..8 {
                let value = structured_biguint(modulus_words, &mut rng).rem(&modulus);
                let encoded = ctx.to_residue(&value);
                // Reducedness is the residue type's invariant now, not a
                // property a caller can inspect; the decode is the observable.
                assert!(ctx.from_residue(&encoded).expect("same context") < modulus);
                assert_eq!(
                    ctx.from_residue(&encoded).expect("same context"),
                    value,
                    "decode∘encode != id"
                );
            }
        }
    }
}

#[test]
fn mul_and_square_match_division_reference() {
    let mut rng = rng();
    for modulus_words in [1usize, 2, 4, 8] {
        for _ in 0..8 {
            let modulus = structured_odd_modulus(modulus_words, &mut rng);
            if modulus.is_one() {
                continue;
            }
            let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

            for _ in 0..8 {
                let a = structured_biguint(modulus_words, &mut rng).rem(&modulus);
                let b = structured_biguint(modulus_words, &mut rng).rem(&modulus);

                let expected = a.mul(&b).rem(&modulus);
                assert_eq!(ctx.mul(&a, &b), expected, "ctx.mul disagrees");
                assert_eq!(
                    ctx.square(&a),
                    a.mul(&a).rem(&modulus),
                    "ctx.square disagrees"
                );

                // The in-domain multiply must commute with encode/decode.
                let product_mont = ctx
                    .mul_residue(&ctx.to_residue(&a), &ctx.to_residue(&b))
                    .expect("same context");
                assert!(ctx.from_residue(&product_mont).expect("same context") < modulus);
                assert_eq!(
                    ctx.from_residue(&product_mont).expect("same context"),
                    expected,
                    "mul_residue disagrees"
                );
                assert_eq!(
                    ctx.square_residue(&ctx.to_residue(&a))
                        .expect("same context"),
                    ctx.mul_residue(&ctx.to_residue(&a), &ctx.to_residue(&a))
                        .expect("same context"),
                    "square_residue disagrees with mul_residue"
                );
            }
        }
    }
}

#[test]
fn squaring_kernel_stresses_carry_chains() {
    // `square_mont` runs the dedicated squaring kernel (mont_sqr: cross terms
    // once, a separate doubling pass, then the diagonal) while `mul_mont` runs
    // the generic multiply (mont_mul: full schoolbook product). These are
    // distinct kernels, so this comparison is a genuine differential test of
    // the squaring path — not, as before, a function against itself. Values
    // with long all-ones runs and isolated high bits push the doubling pass's
    // carries the furthest; compare at several widths, including key sizes.
    let mut rng = rng();
    for modulus_words in [2usize, 4, 8, 16, 32, 64] {
        let modulus = structured_odd_modulus(modulus_words, &mut rng);
        if modulus.is_one() {
            continue;
        }
        let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

        let mut values = vec![
            BigUint::zero(),
            BigUint::one(),
            modulus.sub(&BigUint::one()),
            from_limbs(&vec![u64::MAX; modulus_words]).rem(&modulus),
        ];
        for _ in 0..6 {
            values.push(structured_biguint(modulus_words, &mut rng).rem(&modulus));
        }

        for value in &values {
            let encoded = ctx.to_residue(value);
            assert_eq!(
                ctx.square_residue(&encoded).expect("same context"),
                ctx.mul_residue(&encoded, &encoded).expect("same context"),
                "squaring != multiply-by-self for {value:?} mod {modulus:?}"
            );
        }
    }
}

#[test]
fn pow_at_public_key_sizes() {
    // One full-size spot check per key size the schemes actually use; the
    // reference ladder is quadratic, so counts stay small.
    let mut rng = rng();
    for bits in [256usize, 1024, 2048] {
        let words = bits / 64;
        let modulus = structured_odd_modulus(words, &mut rng);
        let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");
        let base = structured_biguint(words, &mut rng);

        check_pow(&ctx, &base, &BigUint::from_u64(65_537));
        check_pow(&ctx, &base, &structured_biguint(2, &mut rng));
    }
}
