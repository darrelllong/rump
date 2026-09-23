//! Montgomery arithmetic at machine width: one context per odd word modulus,
//! residues that are plain words.
//!
//! Montgomery, *Modular Multiplication Without Trial Division*, Mathematics
//! of Computation 44 (1985). The word-level inverse is computed by Newton's
//! iteration (each step doubles the bits of a 2-adic inverse), a folklore
//! refinement of the Dussé–Kaliski word-level constant.
//!
//! # Why this exists beside [`crate::modular::MontgomeryContext`]
//!
//! The `BigUint` context serves moduli of any width and pays for it in limb
//! loops, allocation, and provenance checks. Inner loops such as a rho walk,
//! an elliptic-curve stage, or a strong-pseudoprime test perform millions of
//! modular multiplications on values of one or two words, where REDC is a
//! handful of register operations. These contexts are that handful, with no
//! heap.
//!
//! # The residue types
//!
//! [`Residue64`] and [`Residue128`] are `Copy` newtypes around the bare
//! word. They carry no context identity, which would double their size, so
//! mixing residues of two contexts is not caught. The types catch the
//! commoner mistake: passing a plain integer where Montgomery form is
//! expected, or reading a Montgomery word back as a value. Neither compiles.

/// Montgomery arithmetic modulo an odd `u64`.
///
/// Construction rejects even and trivial moduli; everything after
/// construction is total.
#[derive(Clone, Copy, Debug)]
pub struct Montgomery64 {
    modulus: u64,
    /// `-modulus⁻¹ mod 2⁶⁴`.
    neg_inverse: u64,
    /// `2¹²⁸ mod modulus`: the conversion constant into the domain.
    r_squared: u64,
    /// `2⁶⁴ mod modulus`: the domain's one.
    one: u64,
}

/// A value in a [`Montgomery64`] domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residue64(u64);

impl Montgomery64 {
    /// The context for an odd modulus greater than one.
    #[must_use]
    pub fn new(modulus: u64) -> Option<Self> {
        if modulus <= 1 || modulus & 1 == 0 {
            return None;
        }
        // Newton in the 2-adics: from a seed correct to 3 bits, each step
        // doubles the correct bits, so five steps clear 64.
        let mut inverse = modulus; // correct mod 2³ for odd modulus
        for _ in 0..5 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(modulus.wrapping_mul(inverse)));
        }
        debug_assert_eq!(modulus.wrapping_mul(inverse), 1);
        // 2⁶⁴ mod modulus. The sum stays below modulus: an odd modulus above
        // one does not divide 2⁶⁴.
        let one = (u64::MAX % modulus) + 1;
        // 2¹²⁸ mod modulus by squaring 2⁶⁴ mod modulus in the plain ring.
        let r_squared = ((u128::from(one) * u128::from(one)) % u128::from(modulus)) as u64;
        Some(Self {
            modulus,
            neg_inverse: inverse.wrapping_neg(),
            r_squared,
            one,
        })
    }

    /// The modulus this context reduces by.
    #[must_use]
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Montgomery reduction of a double-word value: `t·2⁻⁶⁴ mod modulus`.
    #[inline]
    fn redc(&self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.neg_inverse);
        let correction = u128::from(m) * u128::from(self.modulus);
        // `t + correction` can carry past 2¹²⁸, so the halves are summed
        // separately. The low halves cancel by construction of `m` — their
        // sum is exactly 2⁶⁴ when the low word of `t` is nonzero, and zero
        // when it is — so only their carry survives.
        let carry = u128::from(t as u64 != 0);
        let folded = (t >> 64) + (correction >> 64) + carry;
        // One conditional subtraction finishes the reduction: the folded
        // value is below 2·modulus whenever both inputs were below modulus.
        let modulus = u128::from(self.modulus);
        (if folded >= modulus {
            folded - modulus
        } else {
            folded
        }) as u64
    }

    /// `value mod modulus`, carried into the domain.
    #[inline]
    #[must_use]
    pub fn enter(&self, value: u64) -> Residue64 {
        Residue64(self.redc(u128::from(value % self.modulus) * u128::from(self.r_squared)))
    }

    /// The plain integer a residue stands for.
    #[inline]
    #[must_use]
    pub fn exit(&self, value: Residue64) -> u64 {
        self.redc(u128::from(value.0))
    }

    /// The domain's zero.
    #[inline]
    #[must_use]
    pub fn zero(&self) -> Residue64 {
        Residue64(0)
    }

    /// The domain's one.
    #[inline]
    #[must_use]
    pub fn one(&self) -> Residue64 {
        Residue64(self.one)
    }

    /// `a·b` in the domain.
    #[inline]
    #[must_use]
    pub fn mul(&self, a: Residue64, b: Residue64) -> Residue64 {
        Residue64(self.redc(u128::from(a.0) * u128::from(b.0)))
    }

    /// `a²` in the domain.
    #[inline]
    #[must_use]
    pub fn square(&self, a: Residue64) -> Residue64 {
        self.mul(a, a)
    }

    /// `a + b` in the domain.
    #[inline]
    #[must_use]
    pub fn add(&self, a: Residue64, b: Residue64) -> Residue64 {
        let (sum, overflow) = a.0.overflowing_add(b.0);
        Residue64(if overflow || sum >= self.modulus {
            sum.wrapping_sub(self.modulus)
        } else {
            sum
        })
    }

    /// `a − b` in the domain.
    #[inline]
    #[must_use]
    pub fn sub(&self, a: Residue64, b: Residue64) -> Residue64 {
        let (diff, borrow) = a.0.overflowing_sub(b.0);
        Residue64(if borrow {
            diff.wrapping_add(self.modulus)
        } else {
            diff
        })
    }

    /// `base^exponent` in the domain, by binary square-and-multiply.
    ///
    /// Not constant-time, like everything here: the exponent is assumed
    /// public.
    #[must_use]
    pub fn pow(&self, base: Residue64, exponent: u64) -> Residue64 {
        let mut result = self.one();
        let mut square = base;
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.mul(result, square);
            }
            square = self.square(square);
            remaining >>= 1;
        }
        result
    }
}

/// Montgomery arithmetic modulo an odd `u128`.
///
/// The same construction one level up: the double-width products a `u64`
/// context takes from the hardware are assembled here from four word
/// products, which is still loop-free and allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct Montgomery128 {
    modulus: u128,
    /// `-modulus⁻¹ mod 2¹²⁸`.
    neg_inverse: u128,
    /// `2²⁵⁶ mod modulus`.
    r_squared: u128,
    /// `2¹²⁸ mod modulus`.
    one: u128,
}

/// A value in a [`Montgomery128`] domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residue128(u128);

/// `a·b` as a `(high, low)` pair of `u128` halves.
#[inline]
fn wide_mul(a: u128, b: u128) -> (u128, u128) {
    const MASK: u128 = u64::MAX as u128;
    let (a_high, a_low) = (a >> 64, a & MASK);
    let (b_high, b_low) = (b >> 64, b & MASK);
    let low_low = a_low * b_low;
    let low_high = a_low * b_high;
    let high_low = a_high * b_low;
    let high_high = a_high * b_high;
    let mid = (low_low >> 64) + (low_high & MASK) + (high_low & MASK);
    let low = (mid << 64) | (low_low & MASK);
    let high = high_high + (low_high >> 64) + (high_low >> 64) + (mid >> 64);
    (high, low)
}

impl Montgomery128 {
    /// The context for an odd modulus greater than one.
    #[must_use]
    pub fn new(modulus: u128) -> Option<Self> {
        if modulus <= 1 || modulus & 1 == 0 {
            return None;
        }
        // Newton in the 2-adics again; six steps clear 128 bits.
        let mut inverse = modulus;
        for _ in 0..6 {
            inverse = inverse.wrapping_mul(2u128.wrapping_sub(modulus.wrapping_mul(inverse)));
        }
        debug_assert_eq!(modulus.wrapping_mul(inverse), 1);
        // 2¹²⁸ mod modulus, below modulus for the same reason.
        let one = (u128::MAX % modulus) + 1;
        // 2²⁵⁶ mod modulus: square 2¹²⁸ mod modulus with the wide product
        // and reduce the 256-bit square by double-and-add over the high
        // half. Runs once per context; clarity over speed.
        let r_squared = {
            let (high, low) = wide_mul(one, one);
            // Reduce (high·2¹²⁸ + low) mod modulus. Horner over the high
            // half's bits, adding 2¹²⁸ mod modulus — that is, `one` — per set
            // bit, gives high·2¹²⁸ mod modulus; the low half then joins
            // directly.
            let mut accumulator = 0u128;
            for shift in (0..128).rev() {
                accumulator = mod_double(accumulator, modulus);
                if (high >> shift) & 1 == 1 {
                    accumulator = mod_add(accumulator, one, modulus);
                }
            }
            mod_add(accumulator, low % modulus, modulus)
        };
        Some(Self {
            modulus,
            neg_inverse: inverse.wrapping_neg(),
            r_squared,
            one,
        })
    }

    /// The modulus this context reduces by.
    #[must_use]
    pub fn modulus(&self) -> u128 {
        self.modulus
    }

    /// Montgomery reduction of `(high, low)`: `t·2⁻¹²⁸ mod modulus`.
    #[inline]
    fn redc(&self, high: u128, low: u128) -> u128 {
        let m = low.wrapping_mul(self.neg_inverse);
        let (product_high, product_low) = wide_mul(m, self.modulus);
        // The low halves cancel by construction of `m`; only their carry
        // survives into the kept half.
        let (cancelled, carried) = low.overflowing_add(product_low);
        debug_assert_eq!(cancelled, 0);
        let carry = u128::from(carried);
        let (folded, overflow) = high.overflowing_add(product_high);
        let (folded, overflow_carry) = folded.overflowing_add(carry);
        if overflow || overflow_carry || folded >= self.modulus {
            folded.wrapping_sub(self.modulus)
        } else {
            folded
        }
    }

    /// `value mod modulus`, carried into the domain.
    #[inline]
    #[must_use]
    pub fn enter(&self, value: u128) -> Residue128 {
        let (high, low) = wide_mul(value % self.modulus, self.r_squared);
        Residue128(self.redc(high, low))
    }

    /// The plain integer a residue stands for.
    #[inline]
    #[must_use]
    pub fn exit(&self, value: Residue128) -> u128 {
        self.redc(0, value.0)
    }

    /// The domain's zero.
    #[inline]
    #[must_use]
    pub fn zero(&self) -> Residue128 {
        Residue128(0)
    }

    /// The domain's one.
    #[inline]
    #[must_use]
    pub fn one(&self) -> Residue128 {
        Residue128(self.one)
    }

    /// `a·b` in the domain.
    #[inline]
    #[must_use]
    pub fn mul(&self, a: Residue128, b: Residue128) -> Residue128 {
        let (high, low) = wide_mul(a.0, b.0);
        Residue128(self.redc(high, low))
    }

    /// `a²` in the domain.
    #[inline]
    #[must_use]
    pub fn square(&self, a: Residue128) -> Residue128 {
        self.mul(a, a)
    }

    /// `a + b` in the domain.
    #[inline]
    #[must_use]
    pub fn add(&self, a: Residue128, b: Residue128) -> Residue128 {
        let (sum, overflow) = a.0.overflowing_add(b.0);
        Residue128(if overflow || sum >= self.modulus {
            sum.wrapping_sub(self.modulus)
        } else {
            sum
        })
    }

    /// `a − b` in the domain.
    #[inline]
    #[must_use]
    pub fn sub(&self, a: Residue128, b: Residue128) -> Residue128 {
        let (diff, borrow) = a.0.overflowing_sub(b.0);
        Residue128(if borrow {
            diff.wrapping_add(self.modulus)
        } else {
            diff
        })
    }

    /// `base^exponent` in the domain, by binary square-and-multiply.
    #[must_use]
    pub fn pow(&self, base: Residue128, exponent: u128) -> Residue128 {
        let mut result = self.one();
        let mut square = base;
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self.mul(result, square);
            }
            square = self.square(square);
            remaining >>= 1;
        }
        result
    }
}

/// `2a mod modulus` without overflow.
#[inline]
fn mod_double(a: u128, modulus: u128) -> u128 {
    let reduced = a % modulus;
    let room = modulus - reduced;
    if reduced >= room {
        reduced - room
    } else {
        reduced << 1
    }
}

/// `a + b mod modulus` without overflow; the inputs need not be reduced.
#[inline]
fn mod_add(a: u128, b: u128, modulus: u128) -> u128 {
    let b = b % modulus;
    let room = modulus - (a % modulus);
    if b >= room {
        b - room
    } else {
        (a % modulus) + b
    }
}

/// Knuth's MMIX linear congruential generator (*TAOCP* vol. 2, §3.3.4):
/// the multiplier and increment the tests walk the word with.
#[cfg(test)]
const MMIX_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
#[cfg(test)]
const MMIX_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// 2⁶⁴ − 59 and 2¹²⁸ − 159, the largest primes below each power, so a
/// context built on either is a field.
#[cfg(test)]
const PRIME_BELOW_2_64: u64 = u64::MAX - 58;
#[cfg(test)]
const PRIME_BELOW_2_128: u128 = u128::MAX - 158;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BigUint;

    /// Seeds, one per test: arbitrary, fixed so a failure reproduces. The
    /// first is the golden-ratio word, here only a seed.
    const PRODUCT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    const CROSS_WIDTH_SEED: u64 = 1;

    /// Random `(modulus, a, b)` triples checked against a `u128` remainder;
    /// the count is a budget, since one triple costs a division.
    const PRODUCT_SAMPLES: usize = 1_000;
    /// Products checked between the word and wide contexts, on the same
    /// footing.
    const CROSS_WIDTH_SAMPLES: usize = 200;

    /// A second generator for the cross-width test, so its stream is not
    /// the product test's: an LCG modulo 2⁶⁴ with the full period, which
    /// Hull & Dobell give for any odd increment and a multiplier that is 1
    /// modulo 4 (this one is 2⁶² + 1 modulo 4). The multiplier is otherwise
    /// arbitrary; nothing here depends on its lattice quality. The rotation
    /// is stream shaping: the second operand is the state turned so the
    /// pair is not a value with itself.
    const STREAM_MULTIPLIER: u64 = 2_862_933_555_777_941_757;
    const STREAM_INCREMENT: u64 = 3;
    const PARTNER_ROTATION: u32 = 31;

    /// Arbitrary small operands, far from the modulus.
    const SMALL_PAIR: (u64, u64) = (12_345, 67_890);

    /// 2¹⁰⁰ + 277, prime: past a word, so the wide context reduces for real.
    const WIDE_PRIME: u128 = (1 << 100) + 277;
    /// Arbitrary operands above a word, so the wide product's high half is
    /// not zero.
    const WIDE_A: u128 = (1 << 99) + 12_345;
    const WIDE_B: u128 = (1 << 98) + 67_890;

    /// Moduli for the round trip: the two smallest odd ones, F4 = 2¹⁶ + 1,
    /// M31 = 2³¹ − 1, the largest prime below 2⁶⁴, and 2⁶⁴ − 1 itself.
    const ROUND_TRIP_MODULI: [u64; 6] = [3, 5, 65_537, 2_147_483_647, PRIME_BELOW_2_64, u64::MAX];

    #[test]
    fn even_and_trivial_moduli_are_refused() {
        assert!(Montgomery64::new(0).is_none());
        assert!(Montgomery64::new(1).is_none());
        assert!(Montgomery64::new(4).is_none());
        assert!(Montgomery128::new(0).is_none());
        assert!(Montgomery128::new(1).is_none());
        assert!(Montgomery128::new(1 << 100).is_none());
        assert!(Montgomery64::new(2).is_none());
        assert!(Montgomery64::new(u64::MAX - 1).is_none());
        assert!(Montgomery128::new(2).is_none());
        assert!(Montgomery128::new(u128::MAX - 1).is_none());
        assert!(Montgomery64::new(3).is_some());
        assert!(Montgomery64::new(u64::MAX).is_some());
        assert!(Montgomery128::new(3).is_some());
        assert!(Montgomery128::new(u128::MAX).is_some());
    }

    /// Odd moduli near the top of the word, where the sum of two residues
    /// overflows it (and, in the wide context, so does REDC's folded value),
    /// plus small ones where nothing does. Among them: F4 = 2¹⁶ + 1, F5 =
    /// 2³² + 1 (composite), both neighbours of the half-way power, and the
    /// largest prime below each power, so one modulus in each list is a
    /// field.
    const EDGE_MODULI_64: [u64; 8] = [
        3,
        65_537,
        (1 << 32) + 1,
        (1 << 63) - 1,
        (1 << 63) + 1,
        PRIME_BELOW_2_64,
        u64::MAX - 2,
        u64::MAX,
    ];

    const EDGE_MODULI_128: [u128; 9] = [
        3,
        u64::MAX as u128,
        (1 << 64) + 1,
        (1 << 127) - 1,
        (1 << 127) + 1,
        PRIME_BELOW_2_128,
        u128::MAX - 2,
        u128::MAX,
        (u64::MAX as u128) << 64 | 1,
    ];

    /// Operands for a modulus: zero, one, the modulus's own neighbourhood,
    /// and word values with every half all-ones or all-zeros, which drive
    /// the longest carry chains. Unreduced values test `enter`'s reduction.
    fn edge_operands_128(modulus: u128) -> Vec<u128> {
        let word = u128::from(u64::MAX);
        vec![
            0,
            1,
            2,
            modulus / 2,
            modulus / 2 + 1,
            modulus - 2,
            modulus - 1,
            modulus,
            word,
            word + 1,
            word << 64,
            1 << 127,
            u128::MAX - 1,
            u128::MAX,
        ]
    }

    fn big(value: u128) -> BigUint {
        BigUint::from_u128(value)
    }

    #[test]
    fn the_context_constants_are_what_they_claim_at_the_word_edges() {
        for modulus in EDGE_MODULI_64 {
            let context = Montgomery64::new(modulus).expect("odd modulus above one");
            assert_eq!(modulus.wrapping_mul(context.neg_inverse), u64::MAX);
            let one = (1u128 << 64) % u128::from(modulus);
            assert_eq!(u128::from(context.one), one, "2^64 mod {modulus}");
            let r_squared = big(1 << 64)
                .mul(&big(1 << 64))
                .rem(&big(u128::from(modulus)));
            assert_eq!(
                big(u128::from(context.r_squared)),
                r_squared,
                "2^128 mod {modulus}"
            );
        }
        for modulus in EDGE_MODULI_128 {
            let context = Montgomery128::new(modulus).expect("odd modulus above one");
            assert_eq!(modulus.wrapping_mul(context.neg_inverse), u128::MAX);
            let r = big(u128::MAX).add(&big(1));
            assert_eq!(
                big(context.one),
                r.rem(&big(modulus)),
                "2^128 mod {modulus}"
            );
            assert_eq!(
                big(context.r_squared),
                r.mul(&r).rem(&big(modulus)),
                "2^256 mod {modulus}"
            );
        }
    }

    #[test]
    fn wide_products_carry_through_every_half() {
        let word = u128::from(u64::MAX);
        let operands = [
            0,
            1,
            word,
            word + 1,
            word << 64,
            (word << 64) | 1,
            1 << 127,
            u128::MAX - 1,
            u128::MAX,
        ];
        for a in operands {
            for b in operands {
                let (high, low) = wide_mul(a, b);
                let product = big(high)
                    .mul(&big(1 << 64))
                    .mul(&big(1 << 64))
                    .add(&big(low));
                assert_eq!(product, big(a).mul(&big(b)), "{a:#x} * {b:#x}");
            }
        }
    }

    #[test]
    fn the_word_context_is_exact_at_the_top_of_its_range() {
        for modulus in EDGE_MODULI_64 {
            let context = Montgomery64::new(modulus).expect("odd modulus above one");
            let wide_modulus = u128::from(modulus);
            let operands: Vec<u64> = edge_operands_128(wide_modulus)
                .into_iter()
                .map(|value| value as u64)
                .collect();
            for &a in &operands {
                let (x, plain_a) = (context.enter(a), u128::from(a) % wide_modulus);
                assert_eq!(
                    u128::from(context.exit(x)),
                    plain_a,
                    "round trip {a} mod {modulus}"
                );
                let squared = (plain_a * plain_a % wide_modulus) as u64;
                assert_eq!(
                    context.exit(context.square(x)),
                    squared,
                    "{a}^2 mod {modulus}"
                );
                assert_eq!(context.square(x), context.mul(x, x));
                assert_eq!(context.pow(x, 0), context.one());
                assert_eq!(context.pow(x, 3), context.mul(context.square(x), x));
                for &b in &operands {
                    let (y, plain_b) = (context.enter(b), u128::from(b) % wide_modulus);
                    let expect = |value: u128| (value % wide_modulus) as u64;
                    assert_eq!(context.exit(context.mul(x, y)), expect(plain_a * plain_b));
                    assert_eq!(context.exit(context.add(x, y)), expect(plain_a + plain_b));
                    assert_eq!(
                        context.exit(context.sub(x, y)),
                        expect(plain_a + wide_modulus - plain_b),
                        "{a} - {b} mod {modulus}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_wide_context_is_exact_at_the_top_of_its_range() {
        for modulus in EDGE_MODULI_128 {
            let context = Montgomery128::new(modulus).expect("odd modulus above one");
            let big_modulus = big(modulus);
            let reduce = |value: BigUint| value.rem(&big_modulus);
            let operands = edge_operands_128(modulus);
            for &a in &operands {
                let (x, plain_a) = (context.enter(a), reduce(big(a)));
                assert_eq!(
                    big(context.exit(x)),
                    plain_a,
                    "round trip {a} mod {modulus}"
                );
                let squared = reduce(plain_a.mul(&plain_a));
                assert_eq!(
                    big(context.exit(context.square(x))),
                    squared,
                    "{a}^2 mod {modulus}"
                );
                assert_eq!(context.square(x), context.mul(x, x));
                assert_eq!(context.pow(x, 0), context.one());
                assert_eq!(context.pow(x, 3), context.mul(context.square(x), x));
                for &b in &operands {
                    let (y, plain_b) = (context.enter(b), reduce(big(b)));
                    assert_eq!(
                        big(context.exit(context.mul(x, y))),
                        reduce(plain_a.mul(&plain_b)),
                        "{a} * {b} mod {modulus}"
                    );
                    assert_eq!(
                        big(context.exit(context.add(x, y))),
                        reduce(plain_a.add(&plain_b)),
                        "{a} + {b} mod {modulus}"
                    );
                    assert_eq!(
                        big(context.exit(context.sub(x, y))),
                        reduce(plain_a.add(&big_modulus).sub(&plain_b)),
                        "{a} - {b} mod {modulus}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_round_trip_is_the_identity_across_the_range() {
        for modulus in ROUND_TRIP_MODULI {
            let context = Montgomery64::new(modulus).expect("odd moduli above one");
            for &value in &[0u64, 1, 2, modulus / 2, modulus - 1, u64::MAX] {
                let expected = value % modulus;
                assert_eq!(
                    context.exit(context.enter(value)),
                    expected,
                    "round trip failed at {value} mod {modulus}"
                );
            }
        }
    }

    #[test]
    fn multiplication_agrees_with_the_plain_ring() {
        // Deterministic pseudo-random coverage: a linear congruential walk
        // exercises the full width without a random source dependency.
        let mut state = PRODUCT_SEED;
        let mut next = || {
            state = state
                .wrapping_mul(MMIX_MULTIPLIER)
                .wrapping_add(MMIX_INCREMENT);
            state
        };
        for _ in 0..PRODUCT_SAMPLES {
            let modulus = next() | 1;
            if modulus <= 1 {
                continue;
            }
            let context = Montgomery64::new(modulus).expect("odd modulus above one");
            let (a, b) = (next(), next());
            let product = context.exit(context.mul(context.enter(a), context.enter(b)));
            let expected =
                ((u128::from(a % modulus) * u128::from(b % modulus)) % u128::from(modulus)) as u64;
            assert_eq!(product, expected, "{a}·{b} mod {modulus}");
        }
    }

    #[test]
    fn addition_and_subtraction_agree_with_the_plain_ring() {
        let context = Montgomery64::new(u64::MAX).expect("odd");
        let modulus = u128::from(u64::MAX);
        for &(a, b) in &[
            (0u64, 0u64),
            (1, u64::MAX - 1),
            (u64::MAX - 1, u64::MAX - 1),
            SMALL_PAIR,
        ] {
            let sum = context.exit(context.add(context.enter(a), context.enter(b)));
            assert_eq!(u128::from(sum), (u128::from(a) + u128::from(b)) % modulus);
            let difference = context.exit(context.sub(context.enter(a), context.enter(b)));
            assert_eq!(
                u128::from(difference),
                (u128::from(a) + modulus - u128::from(b % u64::MAX)) % modulus
            );
        }
    }

    #[test]
    fn exponentiation_matches_fermat_on_a_prime_modulus() {
        // 2⁶¹ − 1 is prime, so aᵖ⁻¹ ≡ 1 for a ≢ 0.
        let prime = (1u64 << 61) - 1;
        let context = Montgomery64::new(prime).expect("odd prime");
        // The two smallest bases, F4, and the top of the range.
        for &base in &[2u64, 3, 65_537, prime - 2] {
            let result = context.exit(context.pow(context.enter(base), prime - 1));
            assert_eq!(result, 1, "Fermat failed for {base}");
        }
    }

    #[test]
    fn the_wide_context_agrees_with_the_word_context_inside_a_word() {
        let modulus = PRIME_BELOW_2_64;
        let word = Montgomery64::new(modulus).expect("odd");
        let wide = Montgomery128::new(u128::from(modulus)).expect("odd");
        let mut state = CROSS_WIDTH_SEED;
        for _ in 0..CROSS_WIDTH_SAMPLES {
            state = state
                .wrapping_mul(STREAM_MULTIPLIER)
                .wrapping_add(STREAM_INCREMENT);
            let (a, b) = (state, state.rotate_left(PARTNER_ROTATION));
            let narrow = word.exit(word.mul(word.enter(a), word.enter(b)));
            let broad = wide.exit(wide.mul(wide.enter(u128::from(a)), wide.enter(u128::from(b))));
            assert_eq!(u128::from(narrow), broad);
        }
    }

    #[test]
    fn the_wide_context_is_exact_past_the_word_boundary() {
        // Fermat again, and a product checked against schoolbook reduction
        // through wide_mul.
        let prime = WIDE_PRIME;
        let context = Montgomery128::new(prime).expect("odd prime");
        // The two smallest bases and one above a word.
        for &base in &[2u128, 3, (1 << 99) + 1] {
            let result = context.exit(context.pow(context.enter(base), prime - 1));
            assert_eq!(result, 1, "Fermat failed for {base} mod 2^100 + 277");
        }
        let a = WIDE_A;
        let b = WIDE_B;
        let product = context.exit(context.mul(context.enter(a), context.enter(b)));
        // Reduce the 256-bit product by folding the high half bit by bit,
        // as the context builds r², independently of REDC.
        let (high, low) = wide_mul(a % prime, b % prime);
        let mut expected = 0u128;
        for shift in (0..128).rev() {
            expected = mod_double(expected, prime);
            if (high >> shift) & 1 == 1 {
                expected = mod_add(expected, (u128::MAX % prime) + 1, prime);
            }
        }
        let expected = mod_add(expected, low % prime, prime);
        assert_eq!(product, expected);
    }
}

/// The first twelve primes: the bases Sorenson & Webster proved sufficient
/// below 2⁶⁴, and the trial divisors that settle the smallest candidates
/// before any exponentiation.
const MILLER_RABIN_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Whether a `u64` is prime, decided deterministically.
///
/// Strong-pseudoprime tests (Miller, *Riemann's hypothesis and tests for
/// primality*, JCSS 13 (1976); Rabin, *Probabilistic algorithm for testing
/// primality*, J. Number Theory 12 (1980)) to the first twelve primes as
/// bases. Sorenson & Webster, *Strong pseudoprimes to twelve prime bases*,
/// Mathematics of Computation 86 (2017), 985–1003, computed the least
/// composite passing all twelve as 318 665 857 834 031 151 167 461
/// (≈3.19×10²³) — beyond 2⁶⁴ — so within a word the answer is a theorem,
/// not a probability.
///
/// Runs on [`Montgomery64`]: at most twelve word-width exponentiations and
/// no allocation, for callers that test candidates by the million.
#[must_use]
pub fn is_prime_u64(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    for &small in &MILLER_RABIN_BASES {
        if candidate == small {
            return true;
        }
        if candidate.is_multiple_of(small) {
            return false;
        }
    }
    let domain = Montgomery64::new(candidate).expect("odd: even candidates fell to the base 2");
    // candidate − 1 = 2^s · d with d odd.
    let trailing = (candidate - 1).trailing_zeros();
    let odd_part = (candidate - 1) >> trailing;
    let minus_one = domain.sub(domain.zero(), domain.one());
    'bases: for &base in &MILLER_RABIN_BASES {
        let mut x = domain.pow(domain.enter(base), odd_part);
        if x == domain.one() || x == minus_one {
            continue;
        }
        for _ in 1..trailing {
            x = domain.square(x);
            if x == minus_one {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod primality_tests {
    use super::{is_prime_u64, MMIX_INCREMENT, MMIX_MULTIPLIER};

    /// Arbitrary, fixed so a failure reproduces.
    const PRIMALITY_SEED: u64 = 0x1234_5678_9abc_def1;
    /// Odd words tested: about one in 22 odd 64-bit words is prime
    /// (`2/ln 2⁶⁴`), so the sample holds about 90 primes among the
    /// composites, and each reference costs one BPSW test on a word.
    const PRIMALITY_SAMPLES: usize = 2_000;

    #[test]
    fn agrees_with_bpsw_across_a_mixed_sample() {
        let mut state = PRIMALITY_SEED;
        for _ in 0..PRIMALITY_SAMPLES {
            state = state
                .wrapping_mul(MMIX_MULTIPLIER)
                .wrapping_add(MMIX_INCREMENT);
            let candidate = state | 1;
            let reference = rump_bpsw(candidate);
            assert_eq!(
                is_prime_u64(candidate),
                reference,
                "disagreement at {candidate}"
            );
        }
    }

    fn rump_bpsw(candidate: u64) -> bool {
        crate::number_theory_impl::is_probable_prime_bpsw(&crate::BigUint::from_u64(candidate))
    }

    #[test]
    fn the_edges_are_right() {
        assert!(!is_prime_u64(0));
        assert!(!is_prime_u64(1));
        assert!(is_prime_u64(2));
        assert!(is_prime_u64(3));
        assert!(!is_prime_u64(4));
        assert!(is_prime_u64((1 << 61) - 1), "a Mersenne prime");
        assert!(!is_prime_u64(u64::MAX), "3 divides 2^64 - 1");
        // Strong pseudoprimes to base 2 must not slip through.
        assert!(!is_prime_u64(2_047)); // 23 · 89
        assert!(!is_prime_u64(3_215_031_751)); // psp to bases 2,3,5,7
    }
}
