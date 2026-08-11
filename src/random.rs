//! Random sampling of integers and probable primes.
//!
//! rump chooses no entropy source: every function is driven by a
//! caller-supplied [`Rng`], and the output is exactly as good as that
//! source. Cryptographic callers must supply a CSPRNG (the parent
//! cryptography crate bridges its DRBGs here); simulations may supply any
//! deterministic generator. Temporary buffers holding drawn bytes are wiped
//! before release, matching the crate's scrubbing discipline.

use crate::bigint::BigUint;
use crate::number_theory::{gcd, is_probable_prime};

/// A source of random bytes.
///
/// The one-method contract every sampler here needs; implement it by filling
/// `dest` completely. No quality is implied — supplying a cryptographically
/// strong generator is the caller's responsibility when the use demands one.
pub trait Rng {
    /// Fill `dest` with random bytes.
    fn fill_bytes(&mut self, dest: &mut [u8]);
}

/// Draw a random integer in `[0, upper_exclusive)`.
#[must_use]
pub fn random_below<R: Rng + ?Sized>(rng: &mut R, upper_exclusive: &BigUint) -> Option<BigUint> {
    if upper_exclusive.is_zero() {
        return None;
    }

    let bits = upper_exclusive.bits();
    let mut bytes = vec![0u8; bits.div_ceil(8)];
    let excess_bits = bytes.len() * 8 - bits;
    let top_mask = 0xff_u8 >> excess_bits;

    loop {
        rng.fill_bytes(&mut bytes);
        // Rejection sampling from the next power-of-two range. The buffer is
        // big-endian, so masking byte 0 constrains only the most significant
        // partial byte and keeps the candidate below `2^bits`; the loop then
        // retries until the draw lands below `upper_exclusive`. Because the
        // candidate range is the next power of two, the expected retry count
        // stays below 2.
        bytes[0] &= top_mask;
        let candidate = BigUint::from_be_bytes(&bytes);
        crate::scrub::zeroize_slice(bytes.as_mut_slice());
        if candidate < *upper_exclusive {
            return Some(candidate);
        }
    }
}

/// Draw a random integer in `[1, upper_exclusive)`.
#[must_use]
pub fn random_nonzero_below<R: Rng + ?Sized>(
    rng: &mut R,
    upper_exclusive: &BigUint,
) -> Option<BigUint> {
    if upper_exclusive <= &BigUint::one() {
        return None;
    }

    loop {
        let candidate = random_below(rng, upper_exclusive)?;
        if !candidate.is_zero() {
            return Some(candidate);
        }
    }
}

/// Draw a random integer in `[1, upper_exclusive)` that is coprime to `coprime_to`.
///
/// This is the nonce sampler used by schemes such as Paillier that need a
/// fresh random unit modulo `n`: rejection-sample in `[1, upper_exclusive)`
/// until the candidate lands in the multiplicative group with respect to
/// `coprime_to`.
#[must_use]
pub fn random_coprime_below<R: Rng + ?Sized>(
    rng: &mut R,
    upper_exclusive: &BigUint,
    coprime_to: &BigUint,
) -> Option<BigUint> {
    loop {
        let candidate = random_nonzero_below(rng, upper_exclusive)?;
        if gcd(&candidate, coprime_to) == BigUint::one() {
            return Some(candidate);
        }
    }
}

/// Draw a probable prime with the requested bit length.
#[must_use]
pub fn random_probable_prime<R: Rng + ?Sized>(rng: &mut R, bits: usize) -> Option<BigUint> {
    if bits < 2 {
        return None;
    }

    let mut bytes = vec![0u8; bits.div_ceil(8)];
    let top_bit = (bits - 1) % 8;
    let excess_bits = bytes.len() * 8 - bits;
    let top_mask = 0xff_u8 >> excess_bits;
    loop {
        rng.fill_bytes(&mut bytes);
        bytes[0] &= top_mask;
        // Force the requested bit length by setting the top significant bit,
        // then force oddness because every even candidate above 2 is composite.
        bytes[0] |= 1u8 << top_bit;
        let last = bytes.len() - 1;
        bytes[last] |= 1;

        let candidate = BigUint::from_be_bytes(&bytes);
        crate::scrub::zeroize_slice(bytes.as_mut_slice());
        if is_probable_prime(&candidate) {
            return Some(candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        random_below, random_coprime_below, random_nonzero_below, random_probable_prime, Rng,
    };
    use crate::bigint::BigUint;
    use crate::number_theory::{gcd, is_probable_prime};

    /// splitmix64 (Steele, Lea & Flood 2014): deterministic scattered bytes.
    struct SplitMix64 {
        state: u64,
    }

    impl Rng for SplitMix64 {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let word = (z ^ (z >> 31)).to_le_bytes();
                chunk.copy_from_slice(&word[..chunk.len()]);
            }
        }
    }

    struct ZeroRng;

    impl Rng for ZeroRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(0);
        }
    }

    #[test]
    fn random_below_respects_its_bound() {
        let mut rng = SplitMix64 { state: 0x00b5_e550 };
        for bound_bits in [1usize, 7, 63, 64, 65, 200] {
            let mut bound = BigUint::zero();
            bound.set_bit(bound_bits);
            bound = bound.sub_ref(&BigUint::one());
            if bound.is_zero() {
                continue;
            }
            for _ in 0..32 {
                let draw = random_below(&mut rng, &bound).expect("non-zero bound");
                assert!(draw < bound);
            }
        }
        assert_eq!(random_below(&mut rng, &BigUint::zero()), None);
    }

    #[test]
    fn random_nonzero_and_coprime_hold_their_contracts() {
        let mut rng = SplitMix64 { state: 0x0c0f_fee0 };
        let bound = BigUint::from_u64(1_000_003);
        let modulus = BigUint::from_u64(30_030); // 2·3·5·7·11·13
        for _ in 0..64 {
            let nz = random_nonzero_below(&mut rng, &bound).expect("bound > 1");
            assert!(!nz.is_zero() && nz < bound);

            let coprime = random_coprime_below(&mut rng, &bound, &modulus).expect("units exist");
            assert_eq!(gcd(&coprime, &modulus), BigUint::one());
        }
        assert_eq!(random_nonzero_below(&mut rng, &BigUint::one()), None);
    }

    #[test]
    fn random_probable_prime_hits_the_requested_width() {
        let mut rng = SplitMix64 { state: 0x0dea_dbee };
        for bits in [8usize, 64, 96, 128] {
            let p = random_probable_prime(&mut rng, bits).expect("bits >= 2");
            assert_eq!(p.bits(), bits, "prime width");
            assert!(p.is_odd());
            assert!(is_probable_prime(&p));
        }
        assert_eq!(random_probable_prime(&mut rng, 1), None);
    }

    #[test]
    fn degenerate_rng_cannot_stall_bounds_checks() {
        // An all-zeros source can never satisfy `nonzero`, but the bound
        // checks that CAN be answered without entropy still are.
        let mut rng = ZeroRng;
        assert_eq!(random_below(&mut rng, &BigUint::zero()), None);
        assert_eq!(random_nonzero_below(&mut rng, &BigUint::one()), None);
        assert_eq!(random_probable_prime(&mut rng, 0), None);
        assert_eq!(
            random_below(&mut rng, &BigUint::from_u64(5)),
            Some(BigUint::zero())
        );
    }
}
