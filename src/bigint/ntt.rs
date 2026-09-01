//! Exact large-integer multiplication by number-theoretic transform.
//!
//! Limbs are split into base-2^16 digits and convolved modulo two NTT
//! primes.  Their product exceeds every possible raw convolution coefficient
//! at the supported transform lengths, so two-prime CRT recovers each integer
//! coefficient uniquely; an ordinary carry pass then returns to base 2^64.
//! This is the modular-transform form of Schönhage and Strassen's fast integer
//! multiplication (Computing 7 (1971), 281–292), with iterative radix-2
//! Cooley–Tukey transforms.

use super::BigUint;

const DIGIT_BITS: usize = 16;
const DIGITS_PER_LIMB: usize = 64 / DIGIT_BITS;
const DIGIT_MASK: u64 = (1 << DIGIT_BITS) - 1;

// Both primes are c·2^k + 1 and admit the listed primitive root.  The second
// prime sets the common transform ceiling at 2^26.
const PRIME_0: u64 = 2_013_265_921; // 15·2^27 + 1
const ROOT_0: u64 = 31;
const PRIME_1: u64 = 1_811_939_329; // 27·2^26 + 1
const ROOT_1: u64 = 13;
const MAX_TRANSFORM_LEN: usize = 1 << 26;
const PRIME_PRODUCT: u64 = PRIME_0 * PRIME_1;

/// Padded transform length for full-width operands, when supported.
pub(super) fn transform_len(lhs_limbs: usize, rhs_limbs: usize) -> Option<usize> {
    let lhs_digits = lhs_limbs.checked_mul(DIGITS_PER_LIMB)?;
    let rhs_digits = rhs_limbs.checked_mul(DIGITS_PER_LIMB)?;
    let convolution_len = lhs_digits
        .checked_add(rhs_digits)
        .and_then(|sum| sum.checked_sub(1))?;
    convolution_len
        .checked_next_power_of_two()
        .filter(|&len| len <= MAX_TRANSFORM_LEN)
}

/// Multiply two non-zero values through an exact two-prime NTT convolution.
pub(super) fn multiply(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    debug_assert!(!lhs.limbs.is_empty() && !rhs.limbs.is_empty());

    let lhs_digits = significant_digit_len(lhs);
    let rhs_digits = significant_digit_len(rhs);
    let convolution_len = lhs_digits
        .checked_add(rhs_digits)
        .and_then(|sum| sum.checked_sub(1))
        .expect("NTT convolution length fits usize");
    let transform_len = convolution_len
        .checked_next_power_of_two()
        .expect("NTT transform length fits usize");
    assert!(
        transform_len <= MAX_TRANSFORM_LEN,
        "NTT transform exceeds the supported 2^26 coefficients"
    );

    // The maximum coefficient is overlap·(2^16-1)^2.  At the largest
    // supported transform, overlap <= 2^25, while PRIME_0·PRIME_1 > 2^61.
    // Keeping this executable assertion beside the CRT prevents a later base
    // or transform-limit change from silently invalidating exact recovery.
    let coefficient_bound =
        (lhs_digits.min(rhs_digits) as u128) * u128::from(DIGIT_MASK) * u128::from(DIGIT_MASK);
    assert!(coefficient_bound < u128::from(PRIME_PRODUCT));

    let mut left = vec![0u64; transform_len];
    let mut right = vec![0u64; transform_len];
    write_digits(lhs, &mut left[..lhs_digits]);
    write_digits(rhs, &mut right[..rhs_digits]);
    convolve_mod::<PRIME_0, ROOT_0>(&mut left, &mut right);
    let residues_0: Vec<u32> = left[..convolution_len]
        .iter()
        .map(|&value| value as u32)
        .collect();

    left.fill(0);
    right.fill(0);
    write_digits(lhs, &mut left[..lhs_digits]);
    write_digits(rhs, &mut right[..rhs_digits]);
    convolve_mod::<PRIME_1, ROOT_1>(&mut left, &mut right);

    // Reconstruct coefficients and carry directly into packed 64-bit limbs.
    let mut limbs = Vec::with_capacity((convolution_len + DIGITS_PER_LIMB) / 4);
    let mut word = 0u64;
    let mut digit_in_word = 0usize;
    let mut carry = 0u128;
    let mut push_digit = |digit: u64| {
        word |= digit << (DIGIT_BITS * digit_in_word);
        digit_in_word += 1;
        if digit_in_word == DIGITS_PER_LIMB {
            limbs.push(word);
            word = 0;
            digit_in_word = 0;
        }
    };

    for (&residue_0, &residue_1) in residues_0.iter().zip(&left) {
        let coefficient = u128::from(crt_two(u64::from(residue_0), residue_1)) + carry;
        push_digit((coefficient as u64) & DIGIT_MASK);
        carry = coefficient >> DIGIT_BITS;
    }
    while carry != 0 {
        push_digit((carry as u64) & DIGIT_MASK);
        carry >>= DIGIT_BITS;
    }
    if digit_in_word != 0 {
        limbs.push(word);
    }
    BigUint::from_limbs(limbs)
}

fn significant_digit_len(value: &BigUint) -> usize {
    let top = *value
        .limbs
        .last()
        .expect("NTT multiplication receives non-zero operands");
    (value.limbs.len() - 1)
        .checked_mul(DIGITS_PER_LIMB)
        .and_then(|digits| {
            digits.checked_add((64 - top.leading_zeros() as usize).div_ceil(DIGIT_BITS))
        })
        .expect("NTT digit length fits usize")
}

fn write_digits(value: &BigUint, digits: &mut [u64]) {
    let mut index = 0usize;
    for &limb in &value.limbs {
        for shift in (0..64).step_by(DIGIT_BITS) {
            if index == digits.len() {
                return;
            }
            digits[index] = (limb >> shift) & DIGIT_MASK;
            index += 1;
        }
    }
}

/// Recover `x < PRIME_0·PRIME_1` from its two residues.
fn crt_two(residue_0: u64, residue_1: u64) -> u64 {
    // PRIME_0^-1 mod PRIME_1 = -9.  Thus
    // x = residue_0 + PRIME_0·((residue_1-residue_0)·(-9) mod PRIME_1).
    let residue_0_mod_1 = if residue_0 >= PRIME_1 {
        residue_0 - PRIME_1
    } else {
        residue_0
    };
    let delta = if residue_1 >= residue_0_mod_1 {
        residue_1 - residue_0_mod_1
    } else {
        residue_1 + PRIME_1 - residue_0_mod_1
    };
    let negated = (9 * delta) % PRIME_1;
    let multiplier = if negated == 0 { 0 } else { PRIME_1 - negated };
    residue_0 + PRIME_0 * multiplier
}

fn convolve_mod<const MODULUS: u64, const ROOT: u64>(left: &mut [u64], right: &mut [u64]) {
    debug_assert_eq!(left.len(), right.len());
    transform::<MODULUS, ROOT>(left, false);
    transform::<MODULUS, ROOT>(right, false);
    for (lhs, &rhs) in left.iter_mut().zip(right.iter()) {
        *lhs = mul_mod::<MODULUS>(*lhs, rhs);
    }
    transform::<MODULUS, ROOT>(left, true);
}

/// In-place iterative radix-2 Cooley–Tukey transform.
fn transform<const MODULUS: u64, const ROOT: u64>(values: &mut [u64], inverse: bool) {
    debug_assert!(values.len().is_power_of_two());
    debug_assert!((MODULUS - 1).is_multiple_of(values.len() as u64));

    // Bit-reversal permutation, incrementally updating the reversed index.
    let mut reversed = 0usize;
    for index in 1..values.len() {
        let mut bit = values.len() >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut width = 2usize;
    while width <= values.len() {
        let mut root = pow_mod::<MODULUS>(ROOT, (MODULUS - 1) / width as u64);
        if inverse {
            root = pow_mod::<MODULUS>(root, MODULUS - 2);
        }
        for block in values.chunks_exact_mut(width) {
            let (low, high) = block.split_at_mut(width / 2);
            let mut weight = 1u64;
            for (even, odd) in low.iter_mut().zip(high.iter_mut()) {
                let lhs = *even;
                let rhs = mul_mod::<MODULUS>(*odd, weight);
                let sum = lhs + rhs;
                *even = if sum >= MODULUS { sum - MODULUS } else { sum };
                *odd = if lhs >= rhs {
                    lhs - rhs
                } else {
                    lhs + MODULUS - rhs
                };
                weight = mul_mod::<MODULUS>(weight, root);
            }
        }
        width <<= 1;
    }

    if inverse {
        let inverse_len = pow_mod::<MODULUS>(values.len() as u64, MODULUS - 2);
        for value in values {
            *value = mul_mod::<MODULUS>(*value, inverse_len);
        }
    }
}

#[inline]
fn mul_mod<const MODULUS: u64>(lhs: u64, rhs: u64) -> u64 {
    // Both moduli are below 2^31, so the product is below 2^62 and cannot
    // overflow.  Keeping MODULUS const lets the optimizer replace division by
    // each fixed prime with multiplication by a reciprocal.
    (lhs * rhs) % MODULUS
}

fn pow_mod<const MODULUS: u64>(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod::<MODULUS>(result, base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = mul_mod::<MODULUS>(base, base);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_generate_the_required_power_of_two_subgroups() {
        let is_prime_by_trial_division = |candidate: u64| {
            if candidate < 2 || candidate.is_multiple_of(2) {
                return candidate == 2;
            }
            let mut divisor = 3u64;
            while divisor * divisor <= candidate {
                if candidate.is_multiple_of(divisor) {
                    return false;
                }
                divisor += 2;
            }
            true
        };
        assert!(is_prime_by_trial_division(PRIME_0));
        assert!(is_prime_by_trial_division(PRIME_1));
        assert_eq!(PRIME_0 - 1, 15 * (1 << 27));
        assert_eq!(PRIME_1 - 1, 27 * (1 << 26));

        // The distinct prime factors of PRIME_0-1 are 2, 3, 5; those of
        // PRIME_1-1 are 2, 3. Non-unity at each quotient proves full order.
        for factor in [2, 3, 5] {
            assert_ne!(pow_mod::<PRIME_0>(ROOT_0, (PRIME_0 - 1) / factor), 1);
        }
        for factor in [2, 3] {
            assert_ne!(pow_mod::<PRIME_1>(ROOT_1, (PRIME_1 - 1) / factor), 1);
        }
    }

    #[test]
    fn crt_round_trips_boundary_values() {
        for value in [
            0,
            1,
            PRIME_1 - 1,
            PRIME_1,
            PRIME_0,
            PRIME_PRODUCT - 2,
            PRIME_PRODUCT - 1,
        ] {
            assert_eq!(crt_two(value % PRIME_0, value % PRIME_1), value);
        }
    }

    #[test]
    fn transform_round_trip() {
        for len in [1usize, 2, 4, 8, 32, 256] {
            let original: Vec<u64> = (0..len)
                .map(|index| (index as u64 * 1_234_567 + 89) % PRIME_0)
                .collect();
            let mut transformed = original.clone();
            transform::<PRIME_0, ROOT_0>(&mut transformed, false);
            transform::<PRIME_0, ROOT_0>(&mut transformed, true);
            assert_eq!(transformed, original);
        }
    }

    #[test]
    fn transform_length_ceiling_is_exact() {
        // Four base-2^16 digits per limb and two equal operands require just
        // under eight transform coefficients per limb.
        assert_eq!(transform_len(1 << 23, 1 << 23), Some(MAX_TRANSFORM_LEN));
        assert_eq!(transform_len((1 << 23) + 1, (1 << 23) + 1), None);
    }
}
