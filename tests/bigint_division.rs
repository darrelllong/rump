//! Differential and invariant tests for `BigUint` division.
//!
//! `div_rem` implements Knuth's Algorithm D, whose quotient-digit estimate and
//! add-back correction are easy to get subtly wrong in ways that only show up
//! on rare inputs. Every case here is checked two ways:
//!
//! 1. against a deliberately naive bit-at-a-time long division that shares no
//!    code with the implementation, so agreement is evidence rather than a
//!    restatement; and
//! 2. against `q * d + r == n` with `r < d`, the pair of conditions that
//!    uniquely determine Euclidean division.
//!
//! Trial counts scale with the `BIGINT_FUZZ_TRIALS` environment variable so the
//! default `cargo test` stays quick while a soak run can push the same code
//! through orders of magnitude more cases.

use rump::{BigUint, MontgomeryContext};

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

/// Textbook bit-at-a-time long division: shift the remainder left, append the
/// next dividend bit, subtract when the divisor fits. Quadratic and slow, which
/// is the point — it is structurally unlike Algorithm D.
fn div_rem_bitwise(dividend: &BigUint, divisor: &BigUint) -> (BigUint, BigUint) {
    assert!(!divisor.is_zero(), "division by zero");

    let mut quotient = BigUint::zero();
    let mut remainder = BigUint::zero();
    for bit in (0..dividend.bits()).rev() {
        remainder.shl1();
        if dividend.bit(bit) {
            remainder.set_bit(0);
        }
        if remainder >= *divisor {
            remainder.sub_assign_ref(divisor);
            quotient.set_bit(bit);
        }
    }

    (quotient, remainder)
}

fn from_limbs(limbs: &[u64]) -> BigUint {
    let mut bytes = Vec::with_capacity(limbs.len() * 8);
    for &limb in limbs.iter().rev() {
        bytes.extend_from_slice(&limb.to_be_bytes());
    }
    BigUint::from_be_bytes(&bytes)
}

/// Check one division every available way.
fn check(dividend: &BigUint, divisor: &BigUint) {
    let (quotient, remainder) = dividend.div_rem(divisor);

    let (expected_q, expected_r) = div_rem_bitwise(dividend, divisor);
    assert_eq!(
        quotient, expected_q,
        "quotient disagrees with bitwise reference\n  n = {dividend:?}\n  d = {divisor:?}"
    );
    assert_eq!(
        remainder, expected_r,
        "remainder disagrees with bitwise reference\n  n = {dividend:?}\n  d = {divisor:?}"
    );

    assert!(
        remainder < *divisor,
        "remainder not reduced\n  n = {dividend:?}\n  d = {divisor:?}"
    );
    assert_eq!(
        quotient.mul(divisor).add(&remainder),
        *dividend,
        "q * d + r != n\n  n = {dividend:?}\n  d = {divisor:?}"
    );

    // `rem` must agree with the remainder half of `div_rem`.
    assert_eq!(dividend.rem(divisor), remainder);
}

fn trials(default: usize) -> usize {
    std::env::var("BIGINT_FUZZ_TRIALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn rng() -> SplitMix64 {
    SplitMix64::new(0x5a5a_5a5a_5a5a_5a5a)
}

fn random_limb(rng: &mut SplitMix64) -> u64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    u64::from_le_bytes(bytes)
}

/// Draw a limb biased toward the values that break carry, borrow and
/// normalization logic: all zeros, all ones, and the powers of two at the limb
/// boundary. Uniformly random limbs essentially never produce a zero limb in
/// the middle of a number or a divisor needing the maximum D1 shift.
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

#[test]
fn div_rem_matches_bitwise_reference_over_limb_shapes() {
    let mut rng = rng();
    let per_shape = trials(128);

    for dividend_words in 1..=12usize {
        for divisor_words in 1..=dividend_words {
            for _ in 0..per_shape {
                let dividend = structured_biguint(dividend_words, &mut rng);
                let divisor = structured_biguint(divisor_words, &mut rng);
                if divisor.is_zero() {
                    continue;
                }
                check(&dividend, &divisor);
            }
        }
    }
}

#[test]
fn div_rem_matches_bitwise_reference_at_key_sizes() {
    // The sizes the public-key layer actually divides at, including the
    // reduce-a-product shape (`2n` bits mod `n` bits) that `mod_mul` produces.
    let mut rng = rng();
    let per_shape = trials(24);

    for bits in [256usize, 512, 1024, 2048, 4096] {
        let words = bits / 64;
        for divisor_words in [1, 2, words / 2, words - 1, words] {
            for _ in 0..per_shape {
                let dividend = structured_biguint(2 * words, &mut rng);
                let divisor = structured_biguint(divisor_words, &mut rng);
                if divisor.is_zero() {
                    continue;
                }
                check(&dividend, &divisor);
            }
        }
    }
}

#[test]
fn div_rem_boundary_relationships() {
    // Values sitting exactly on a quotient boundary: one below a multiple of
    // the divisor, exactly on it, and one above. These are where an
    // off-by-one in the estimate or a missed add-back surfaces.
    let mut rng = rng();
    let per_shape = trials(32);

    for divisor_words in 1..=8usize {
        for _ in 0..per_shape {
            let divisor = structured_biguint(divisor_words, &mut rng);
            if divisor.is_zero() {
                continue;
            }

            for multiplier_words in 1..=4usize {
                let multiplier = structured_biguint(multiplier_words, &mut rng);
                if multiplier.is_zero() {
                    continue;
                }

                let exact = multiplier.mul(&divisor);
                check(&exact, &divisor);
                check(&exact.add(&BigUint::one()), &divisor);
                // `k * d - 1` is the add-back case for multi-limb divisors.
                check(&exact.sub(&BigUint::one()), &divisor);
            }
        }
    }
}

#[test]
fn div_rem_normalization_shift_extremes() {
    // The D1 normalization shift is `divisor.top_limb.leading_zeros()`. Pin
    // both ends: a top limb of 1 needs the maximum 63-bit shift, and a top limb
    // with bit 63 already set needs none.
    let mut rng = rng();
    let per_shape = trials(64);

    for top in [1u64, 2, 3, u64::MAX, 1 << 63, (1 << 63) | 1, 1 << 62] {
        for divisor_words in 1..=6usize {
            for dividend_words in divisor_words..=10usize {
                for _ in 0..per_shape {
                    let mut divisor_limbs: Vec<u64> = (0..divisor_words)
                        .map(|_| structured_limb(&mut rng))
                        .collect();
                    divisor_limbs[divisor_words - 1] = top;
                    let divisor = from_limbs(&divisor_limbs);

                    let dividend = structured_biguint(dividend_words, &mut rng);
                    check(&dividend, &divisor);
                }
            }
        }
    }
}

#[test]
fn div_rem_worst_case_quotient_estimates() {
    // Algorithm D's D3 estimate and D6 add-back overlap: D3 pulls an estimate
    // that is at most `q + 2` down to `q + 1`, and D6 removes the last one. So
    // random input cannot distinguish a working D3 from a missing one — an
    // estimate that is exactly one too high is repaired either way. Only two
    // situations separate them, and both need constructed input:
    //
    //   * the two-limb estimate overshoots by two, which D6 alone cannot fix;
    //   * the estimate reaches `2^64` and the divisor's second limb is small
    //     enough that the `v[n-2]` test does not pull it back under, so the
    //     quotient limb silently truncates.
    //
    // `dividend = (divisor << 64k) - 1` drives both. It forces the maximal
    // quotient limb `2^64 - 1` against the maximal remainder `divisor - 1`,
    // which is precisely where the estimate runs hottest.
    // Top limbs from the minimum normalized value (where the estimate runs
    // highest) up to all-ones.
    const TOPS: [u64; 5] = [
        1 << 63,
        (1 << 63) | 1,
        (1 << 63) | (1 << 62),
        u64::MAX - 1,
        u64::MAX,
    ];
    // Low-limb patterns. All-ones maximizes the part of the divisor the
    // two-limb estimate cannot see (overshoot by two); zeros in the second
    // limb neuter the `v[n-2]` test (truncation at `2^64`).
    let patterns: [fn(usize, usize) -> u64; 6] = [
        |_, _| 0,
        |_, _| u64::MAX,
        |i, _| if i % 2 == 0 { 0 } else { u64::MAX },
        |i, _| u64::from(i == 0),
        |i, n| u64::from(i + 2 == n),
        |i, _| if i % 2 == 0 { u64::MAX } else { 1 },
    ];

    for words in 2..=6usize {
        for top in TOPS {
            for pattern in patterns {
                let mut limbs: Vec<u64> = (0..words).map(|i| pattern(i, words)).collect();
                limbs[words - 1] = top;
                let divisor = from_limbs(&limbs);
                if divisor.is_zero() {
                    continue;
                }

                for shift_limbs in 1..=3usize {
                    let mut dividend = divisor.clone();
                    dividend.shl_bits(64 * shift_limbs);

                    // `(d << 64k) - 1`, `- 2`, and the exact multiple: the
                    // boundary either side of the hardest quotient limb.
                    check(&dividend.sub(&BigUint::one()), &divisor);
                    check(&dividend.sub(&BigUint::from_u64(2)), &divisor);
                    check(&dividend, &divisor);
                    check(&dividend.add(&BigUint::one()), &divisor);

                    // Same shape with the divisor shifted off normalization, so
                    // the D1 shift is non-zero and the estimate is recomputed
                    // against shifted limbs.
                    for offset in [1usize, 7, 31, 63] {
                        let mut unnormalized = divisor.clone();
                        for _ in 0..offset {
                            unnormalized.shr1();
                        }
                        if unnormalized.is_zero() {
                            continue;
                        }
                        let mut scaled = unnormalized.clone();
                        scaled.shl_bits(64 * shift_limbs);
                        check(&scaled.sub(&BigUint::one()), &unnormalized);
                    }
                }
            }
        }
    }
}

#[test]
fn div_rem_estimate_overshoot_by_two() {
    // The one input family where Algorithm D's raw two-limb estimate is TWO
    // over the true quotient digit, so the D6 add-back alone cannot save a
    // broken D3 correction loop. Take a divisor `v = [v0, d, .., d]` (little
    // endian) with `d >= 2^63` and small `v0 > 0`; then `r = v - 1` shares its
    // top two limbs `(d, d)` with `v`, and the division window `r * b + u0`
    // yields the raw estimate `floor((d*b + d) / d) = b + 1` against a true
    // digit of `b - 1`.
    //
    // `dividend = (v + r) * b + u0 = v*b + (r*b + u0)` reaches that window on
    // its final quotient digit. Verified against Python bigints, and verified
    // to fail when the D3 correction loop is removed (mutation testing); on
    // random input a missing correction is indistinguishable from a working
    // one, because overshoot by one is repaired by D6 either way.
    let b = {
        let mut b = BigUint::zero();
        b.set_bit(64);
        b
    };
    for d in [
        1u64 << 63,
        (1 << 63) | 1,
        (1 << 63) | (1 << 62),
        u64::MAX - 1,
        u64::MAX,
    ] {
        for v0 in [1u64, 2, 5, u64::MAX] {
            for words in 3..=6usize {
                let mut limbs = vec![d; words];
                limbs[0] = v0;
                let divisor = from_limbs(&limbs);
                let r = divisor.sub(&BigUint::one());

                for u0 in [BigUint::zero(), BigUint::one(), BigUint::from_u64(u64::MAX)] {
                    let dividend = divisor.add(&r).mul(&b).add(&u0);
                    check(&dividend, &divisor);
                }
            }
        }
    }
}

#[test]
fn div_rem_powers_of_two() {
    // Sparse operands: a single set bit exercises the carry chains with almost
    // every limb zero, which structured random draws still rarely produce.
    for dividend_bit in [0usize, 1, 63, 64, 65, 127, 128, 255, 256, 511, 512, 1023] {
        for divisor_bit in [0usize, 1, 63, 64, 65, 127, 128, 255, 256] {
            let mut dividend = BigUint::zero();
            dividend.set_bit(dividend_bit);
            let mut divisor = BigUint::zero();
            divisor.set_bit(divisor_bit);
            check(&dividend, &divisor);

            // And the same shifted off the boundary in both directions.
            check(&dividend.add(&BigUint::one()), &divisor);
            if divisor_bit > 0 {
                check(&dividend, &divisor.sub(&BigUint::one()));
            }
        }
    }
}

#[test]
fn div_rem_degenerate_inputs() {
    let one = BigUint::one();
    let big = BigUint::from_be_bytes(&[0xFF; 64]);

    check(&BigUint::zero(), &one);
    check(&BigUint::zero(), &big);
    check(&one, &one);
    check(&big, &one);
    check(&big, &big);
    check(&big, &big.sub(&one));
    check(&big.sub(&one), &big);

    // Divisor above dividend takes the early exit; quotient must be zero and
    // the remainder must be the dividend untouched.
    let (quotient, remainder) = one.div_rem(&big);
    assert!(quotient.is_zero());
    assert_eq!(remainder, one);
}

#[test]
#[should_panic(expected = "division by zero")]
fn div_rem_rejects_zero_divisor() {
    let _ = BigUint::from_u64(7).div_rem(&BigUint::zero());
}

#[test]
fn mod_mul_matches_reference_and_montgomery() {
    // `mod_mul` is now multiply-then-reduce for every modulus parity, so it
    // must agree with an independent reduction and, for odd moduli, with the
    // reusable Montgomery context it used to build internally.
    let mut rng = rng();
    let per_shape = trials(64);

    for words in [1usize, 2, 4, 8, 16, 32] {
        for _ in 0..per_shape {
            let lhs = structured_biguint(words, &mut rng);
            let rhs = structured_biguint(words, &mut rng);
            let modulus = structured_biguint(words, &mut rng);
            if modulus.is_zero() || modulus.is_one() {
                continue;
            }

            let product = BigUint::mod_mul(&lhs, &rhs, &modulus);
            let (_, expected) = div_rem_bitwise(&lhs.mul(&rhs), &modulus);
            assert_eq!(product, expected, "mod_mul disagrees with reference");
            assert!(product < modulus);

            if modulus.is_odd() {
                let ctx = MontgomeryContext::new(&modulus).expect("odd modulus builds a context");
                assert_eq!(
                    product,
                    ctx.mul(&lhs, &rhs),
                    "mod_mul disagrees with Montgomery context"
                );
            }
        }
    }
}
