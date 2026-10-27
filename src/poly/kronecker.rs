//! Kronecker substitution: a product of polynomials as one product of
//! integers.
//!
//! A polynomial `Σ aᵢxⁱ` with coefficients below `2^w` evaluated at `x = 2^s`
//! is an integer whose binary digits of width `s` are the coefficients, and
//! the product of two such integers is the product polynomial evaluated at
//! `2^s`, whose coefficients `Σ_{i+j=k} aᵢbⱼ` are read back off its digits
//! as long as none reaches `2^s`: `s ≥ wₐ + w_b + ⌊log₂ min(dₐ, d_b)⌋ + 1`
//! for operands of `dₐ` and `d_b` terms, a coefficient of the product being
//! a sum of at most `min(dₐ, d_b)` products (Kronecker, *Grundzüge einer
//! arithmetischen Theorie der algebraischen Grössen*, J. reine angew. Math.
//! 92 (1882), 1–122, §4; the modern account is von zur Gathen & Gerhard,
//! *Modern Computer Algebra*, §8.4). Signed coefficients take a bit more:
//! with `|cₖ| < 2^{s−1}` the digit at `k` is `cₖ` or `cₖ + 2^s` less a
//! borrow, so a digit at or past `2^{s−1}` is a negative coefficient and
//! lends one to the next ([`unpack_signed`]).
//!
//! The substitution pays where integer multiplication is quasi-linear: one
//! product of `d·s` bits then costs about `d` times a coefficient product,
//! against the `d²` coefficient products of the convolution. Under a
//! quadratic or Karatsuba kernel it costs more, so it is taken only where
//! the packed operands would go to the transform
//! ([`BigUint::transform_admits`]). The slot is whole limbs, so packing and
//! unpacking are copies.

use crate::bigint::{BigInt, BigUint, Sign};

/// Whether a product of polynomials of `a` and `b` terms, the coefficients
/// of each at most `a_bits` and `b_bits` wide, is cheaper by substitution:
/// the packed operands would multiply by the transform on one thread.
/// `signed` asks a bit more of a slot.
pub(super) fn pays(a: usize, a_bits: usize, b: usize, b_bits: usize, signed: bool) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let slot = slot_limbs(a_bits, b_bits, a.min(b), signed);
    let (short, long) = (a.min(b) * slot, a.max(b) * slot);
    BigUint::transform_admits(short, long)
}

/// The slot, in limbs: room for a coefficient of the product, the sum of
/// `terms` products of the operands' widths, and a sign bit if signed.
fn slot_limbs(a_bits: usize, b_bits: usize, terms: usize, signed: bool) -> usize {
    let log_terms = usize::BITS - terms.max(1).leading_zeros();
    (a_bits + b_bits + log_terms as usize + usize::from(signed)).div_ceil(64)
}

/// The convolution of `a` and `b`, coefficients in `ℕ`, by substitution.
pub(super) fn convolve(a: &[BigUint], b: &[BigUint]) -> Vec<BigUint> {
    let slot = slot_limbs(widest(a), widest(b), a.len().min(b.len()), false);
    let product = pack(a, slot).mul(&pack(b, slot));
    unpack(&product, slot, a.len() + b.len() - 1)
}

/// `a ⋆ a`, by one square of the packed value.
pub(super) fn square(a: &[BigUint]) -> Vec<BigUint> {
    let slot = slot_limbs(widest(a), widest(a), a.len(), false);
    let product = pack(a, slot).square();
    unpack(&product, slot, 2 * a.len() - 1)
}

/// The convolution of `a` and `b`, coefficients in `ℤ`, by substitution.
pub(super) fn convolve_signed(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let widest_signed =
        |side: &[BigInt]| side.iter().map(|c| c.magnitude().bits()).max().unwrap_or(0);
    let slot = slot_limbs(
        widest_signed(a),
        widest_signed(b),
        a.len().min(b.len()),
        true,
    );
    let product = pack_signed(a, slot).mul(&pack_signed(b, slot));
    unpack_signed(&product, slot, a.len() + b.len() - 1)
}

fn widest(side: &[BigUint]) -> usize {
    side.iter().map(BigUint::bits).max().unwrap_or(0)
}

/// `Σ aᵢ·2^{64·slot·i}`: each coefficient's limbs at its slot.
fn pack(coefficients: &[BigUint], slot: usize) -> BigUint {
    let mut limbs = vec![0u64; coefficients.len() * slot];
    for (i, coefficient) in coefficients.iter().enumerate() {
        let source = coefficient.limbs();
        debug_assert!(source.len() <= slot, "a coefficient fits its slot");
        limbs[i * slot..i * slot + source.len()].copy_from_slice(source);
    }
    BigUint::from_limbs(limbs)
}

/// The first `terms` digits of width `slot` limbs.
fn unpack(value: &BigUint, slot: usize, terms: usize) -> Vec<BigUint> {
    let limbs = value.limbs();
    (0..terms)
        .map(|k| {
            let start = (k * slot).min(limbs.len());
            let end = ((k + 1) * slot).min(limbs.len());
            BigUint::from_limbs(limbs[start..end].to_vec())
        })
        .collect()
}

/// The non-negative coefficients packed less the negative ones packed.
fn pack_signed(coefficients: &[BigInt], slot: usize) -> BigInt {
    let part = |sign: Sign| {
        let magnitudes: Vec<BigUint> = coefficients
            .iter()
            .map(|c| {
                if c.sign() == sign {
                    c.magnitude().clone()
                } else {
                    BigUint::zero()
                }
            })
            .collect();
        BigInt::from_biguint(pack(&magnitudes, slot))
    };
    part(Sign::Positive).sub(&part(Sign::Negative))
}

/// The coefficients of a product packed with a sign bit to spare in each
/// slot: a digit at or past `2^{s−1}` is `cₖ + 2^s`, a negative coefficient
/// that lent one to the digit above, which is repaid before that digit is
/// read. A negative product is unpacked as its negation, coefficient by
/// coefficient.
fn unpack_signed(value: &BigInt, slot: usize, terms: usize) -> Vec<BigInt> {
    let negative = value.sign() == Sign::Negative;
    let limbs = value.magnitude().limbs();
    let mut out = Vec::with_capacity(terms);
    let mut borrow = 0u64;
    for k in 0..terms {
        let start = (k * slot).min(limbs.len());
        let end = ((k + 1) * slot).min(limbs.len());
        let mut digit = vec![0u64; slot];
        digit[..end - start].copy_from_slice(&limbs[start..end]);
        // Repay the borrow: one added to the digit, carried through it.
        let mut carry = borrow;
        for limb in digit.iter_mut() {
            if carry == 0 {
                break;
            }
            let (sum, overflowed) = limb.overflowing_add(carry);
            *limb = sum;
            carry = u64::from(overflowed);
        }
        // A carry out of the slot is a digit of exactly 2^s: a zero
        // coefficient that lends again.
        let (magnitude, is_negative) = if carry != 0 {
            borrow = 1;
            (BigUint::zero(), false)
        } else if digit[slot - 1] >> 63 == 1 {
            // 2^s − digit, the two's complement within the slot.
            let mut complement = 1u64;
            for limb in digit.iter_mut() {
                let (sum, overflowed) = (!*limb).overflowing_add(complement);
                *limb = sum;
                complement = u64::from(overflowed);
            }
            borrow = 1;
            (BigUint::from_limbs(digit), true)
        } else {
            borrow = 0;
            (BigUint::from_limbs(digit), false)
        };
        let coefficient = BigInt::from_biguint(magnitude);
        out.push(if is_negative != negative && !coefficient.is_zero() {
            coefficient.negated()
        } else {
            coefficient
        });
    }
    debug_assert_eq!(borrow, 0, "the top coefficient is non-negative");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }

    fn value(limbs: usize, state: &mut u64) -> BigUint {
        BigUint::from_limbs((0..limbs).map(|_| draw(state)).collect())
    }

    /// Against the coefficient convolution, at widths and degrees including
    /// zero coefficients, one-term operands and unequal degrees.
    #[test]
    fn substitution_matches_the_convolution() {
        let mut state = 0x4d59_5ad5_9c41_8a71u64;
        for (da, db, limbs) in [(1, 1, 3), (5, 5, 7), (5, 3, 40), (2, 9, 17), (5, 5, 300)] {
            let mut a: Vec<BigUint> = (0..da).map(|_| value(limbs, &mut state)).collect();
            let b: Vec<BigUint> = (0..db).map(|_| value(limbs - 1, &mut state)).collect();
            a[0] = BigUint::zero();
            let plain: Vec<BigUint> = (0..da + db - 1)
                .map(|k| {
                    let mut sum = BigUint::zero();
                    for i in 0..da {
                        if k >= i && k - i < db {
                            sum = sum.add(&a[i].mul(&b[k - i]));
                        }
                    }
                    sum
                })
                .collect();
            assert_eq!(convolve(&a, &b), plain, "{da} by {db} at {limbs} limbs");
            if da == db {
                let plain_square: Vec<BigUint> = (0..2 * da - 1)
                    .map(|k| {
                        let mut sum = BigUint::zero();
                        for i in 0..da {
                            if k >= i && k - i < da {
                                sum = sum.add(&a[i].mul(&a[k - i]));
                            }
                        }
                        sum
                    })
                    .collect();
                assert_eq!(square(&a), plain_square, "square of {da} at {limbs} limbs");
            }
        }
    }

    /// Signed coefficients in eight sign patterns over the terms, with
    /// coefficients that cancel to zero and products that come out
    /// negative; then slots of one limb filled to the bound, a zero top
    /// coefficient, and the substitution through the transform where the
    /// dispatch takes it.
    #[test]
    fn signed_substitution_matches_the_convolution() {
        let mut state = 0x7f4a_7c15_9e37_79b9u64;
        for (da, db, limbs) in [
            (1, 1, 2),
            (2, 2, 1),
            (5, 5, 6),
            (5, 3, 33),
            (4, 6, 9),
            (5, 5, 200),
        ] {
            for pattern in 0..8u64 {
                let signed = |i: usize, state: &mut u64| {
                    let magnitude = value(limbs, state);
                    let negative = (pattern >> (i % 3)) & 1 == 1;
                    let v = BigInt::from_biguint(magnitude);
                    if negative {
                        v.negated()
                    } else {
                        v
                    }
                };
                let mut a: Vec<BigInt> = (0..da).map(|i| signed(i, &mut state)).collect();
                let mut b: Vec<BigInt> = (0..db).map(|i| signed(i + 1, &mut state)).collect();
                if pattern == 7 {
                    // A product with a zero coefficient: (x − c)(x + c).
                    a = vec![a[0].clone(), BigInt::one()];
                    b = vec![a[0].negated(), BigInt::one()];
                }
                let plain: Vec<BigInt> = (0..a.len() + b.len() - 1)
                    .map(|k| {
                        let mut sum = BigInt::zero();
                        for i in 0..a.len() {
                            if k >= i && k - i < b.len() {
                                sum = sum.add(&a[i].mul(&b[k - i]));
                            }
                        }
                        sum
                    })
                    .collect();
                assert_eq!(
                    convolve_signed(&a, &b),
                    plain,
                    "{da} by {db} at {limbs} limbs, pattern {pattern}"
                );
            }
        }
        // One limb a slot, every coefficient at its bound: 30 bits a side,
        // two bits for the sum of two terms, a sign bit, 63 bits.
        let top = BigInt::from_biguint(BigUint::from_u64((1 << 30) - 1));
        let a = vec![top.clone(), top.negated()];
        let b = vec![top.negated(), top.clone(), BigInt::zero()];
        assert_eq!(slot_limbs(30, 30, 2, true), 1);
        let square = top.mul(&top);
        let plain = vec![
            square.negated(),
            square.add(&square),
            square.negated(),
            BigInt::zero(),
        ];
        assert_eq!(convolve_signed(&a, &b), plain);
        // Through the dispatch: wide enough that the packed product is the
        // transform's, against the coefficient convolution.
        let mut state = 0x1234_5678_9abc_def0u64;
        let wide: Vec<BigInt> = (0..5)
            .map(|i| {
                let v = BigInt::from_biguint(value(1_200, &mut state));
                if i % 2 == 1 {
                    v.negated()
                } else {
                    v
                }
            })
            .collect();
        let widest = wide.iter().map(|c| c.magnitude().bits()).max().unwrap();
        assert!(
            pays(5, widest, 5, widest, true),
            "five 1 200-limb coefficients pack past the transform's crossover"
        );
        let poly = crate::polynomial::PolyZ::new(wide.clone());
        let product = poly.mul(&poly);
        let plain: Vec<BigInt> = (0..9)
            .map(|k| {
                let mut sum = BigInt::zero();
                for i in 0..5 {
                    if k >= i && k - i < 5 {
                        sum = sum.add(&wide[i].mul(&wide[k - i]));
                    }
                }
                sum
            })
            .collect();
        assert_eq!(product.coefficients(), &plain[..]);
    }

    #[test]
    fn slots_hold_the_products_sums() {
        // Five terms of 64 bits each side: 128 bits of product, 3 of sum.
        assert_eq!(slot_limbs(64, 64, 5, false), 3);
        assert_eq!(slot_limbs(64, 64, 5, true), 3);
        assert_eq!(slot_limbs(62, 62, 5, true), 2);
        assert_eq!(slot_limbs(1, 1, 1, false), 1);
    }
}
