//! Random sampling of integers and probable primes.
//!
//! rump chooses no entropy source: every function is driven by a
//! caller-supplied [`Rng`], and the output is exactly as good as that
//! source. Cryptographic callers must supply a CSPRNG (the parent
//! cryptography crate bridges its DRBGs here); simulations may supply any
//! deterministic generator. Temporary buffers holding drawn bytes are wiped
//! before release, matching the crate's scrubbing discipline.
//!
//! Every sampler here is a rejection sampler: it draws from an enclosing set
//! that is easy to sample and discards draws outside the target set. This
//! buys exact uniformity — no modular folding, so no bias toward the low end
//! of the range — at the cost of a draw count that is a random variable
//! rather than a bound. Termination is therefore a probabilistic property of
//! the supplied generator, not a guarantee of the code: a generator whose
//! output is constant, or confined to the rejected region, makes these
//! functions loop indefinitely. Only degenerate *arguments* are reported as
//! `None`; a degenerate *generator* is the caller's to avoid.

use crate::bigint::BigUint;
use crate::number_theory::{gcd, is_probable_prime};

/// A source of random bytes.
///
/// The one-method contract every sampler here needs; implement it by filling
/// `dest` completely. No quality is implied — supplying a cryptographically
/// strong generator is the caller's responsibility when the use demands one.
/// The samplers assume only that successive fills are independent draws;
/// nothing in this module inspects, reseeds, or forks the generator.
pub trait Rng {
    /// Fill `dest` with random bytes.
    ///
    /// Every byte of `dest` must be written. An implementation that leaves
    /// part of the buffer untouched leaves the previous draw's bytes in
    /// place — the samplers reuse one buffer across rejection rounds — and
    /// silently narrows the sampled range.
    fn fill_bytes(&mut self, dest: &mut [u8]);
}

/// Draw a random integer in `[0, upper_exclusive)`, uniformly, or `None` when
/// `upper_exclusive` is zero and the range is empty.
///
/// Uniformity by rejection rather than by reduction: taking a wide draw
/// modulo `upper_exclusive` would over-represent the low residues whenever
/// the bound is not a power of two, so the sample is instead drawn from the
/// enclosing power-of-two range `[0, 2^bits)` — `bits` being the bit length
/// of the bound — and any draw at or above the bound is discarded (Knuth,
/// *TAOCP* vol. 2, §3.4.1, the unbiased range-reduction discussed there).
/// Every accepted value is equally likely because every candidate was.
///
/// Mechanically: one big-endian byte buffer of `ceil(bits / 8)` bytes is
/// filled and its leading byte masked so no bit at index `bits` or above
/// survives, so the candidate is uniform on `[0, 2^bits)`. Because `bits` is
/// the bound's own bit length the bound is at least `2^(bits−1)`, so the
/// acceptance probability is at least one half and the expected number of
/// draws is at most two, with equality exactly when the bound is a power of
/// two. The buffer is scrubbed after each candidate is decoded.
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
        // Rejection sampling from the enclosing power-of-two range. The
        // buffer is big-endian, so masking byte 0 constrains only the most
        // significant partial byte and keeps the candidate below `2^bits`;
        // the loop then retries until the draw lands below `upper_exclusive`.
        // Since `2^bits` is at most twice the bound, the expected number of
        // draws is at most 2.
        bytes[0] &= top_mask;
        let candidate = BigUint::from_be_bytes(&bytes);
        crate::scrub::zeroize_slice(bytes.as_mut_slice());
        if candidate < *upper_exclusive {
            return Some(candidate);
        }
    }
}

/// Draw a random integer in `[1, upper_exclusive)`, uniformly, or `None` when
/// `upper_exclusive` is at most one and the range is empty.
///
/// Zero is the only value [`random_below`] can produce that this range
/// excludes, so the draw is repeated until a non-zero candidate appears.
/// Conditioning a uniform draw on a subset leaves it uniform on that subset,
/// so the result is uniform on `[1, upper_exclusive)`; the rejected mass is
/// one value out of at least two, so the expected number of draws is at most
/// two.
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

/// Draw a random integer in `[1, upper_exclusive)` that is coprime to
/// `coprime_to`, uniformly over that set.
///
/// This is the shape needed to draw a fresh random unit modulo `n`: pass
/// `n` as both bound and modulus and the result is a uniform element of
/// `(Z/nZ)*`. Rejection-sample in `[1, upper_exclusive)` — each candidate
/// uniform by [`random_nonzero_below`] — and keep the first whose gcd with
/// `coprime_to` is one; conditioning a uniform draw on the coprimality
/// predicate leaves it uniform on the coprime residues.
///
/// Returns `None` in two cases, and only these:
///
/// - `upper_exclusive <= 1`, propagated from [`random_nonzero_below`]: the
///   candidate range is empty.
/// - `coprime_to == 0`. Since `gcd(a, 0) = a`, the only integer coprime to
///   zero is 1, so the predicate stops being a filter on residues and becomes
///   an equality test against a single value. The loop would then accept with
///   probability `1 / (upper_exclusive − 1)` per draw — an expected draw count
///   equal to the bound, growing without limit with the operand, for an answer
///   that is the constant 1 whenever it arrives. Reporting the degenerate
///   argument is the correct response, not sampling for a foregone conclusion.
///
/// Whenever `upper_exclusive >= 2` and `coprime_to != 0` a solution exists —
/// 1 is in range and coprime to everything — so the loop terminates almost
/// surely under any generator that can reach it.
#[must_use]
pub fn random_coprime_below<R: Rng + ?Sized>(
    rng: &mut R,
    upper_exclusive: &BigUint,
    coprime_to: &BigUint,
) -> Option<BigUint> {
    // The only integer coprime to 0 is 1 (`gcd(a, 0) = a`), so there is no
    // meaningful random coprime draw: rejection sampling would run for an
    // expected `upper_exclusive` draws to return a constant. Report the
    // degenerate input as no result.
    if coprime_to.is_zero() {
        return None;
    }
    loop {
        let candidate = random_nonzero_below(rng, upper_exclusive)?;
        if gcd(&candidate, coprime_to) == BigUint::one() {
            return Some(candidate);
        }
    }
}

/// Draw a probable prime of exactly `bits` significant bits, or `None` when
/// `bits < 2` and no prime of that width exists (the only value of bit length
/// one is 1).
///
/// Random search in the shape of *Handbook of Applied Cryptography*,
/// Algorithm 4.44, "Random search for a prime using the Miller-Rabin test":
/// draw a candidate of the requested width, screen it, repeat. Each round
/// fills a big-endian buffer of `ceil(bits / 8)` bytes and forces the
/// candidate into the search space with three bit operations — mask the
/// leading byte down to bit `bits − 1`, set that bit so the width is exactly
/// `bits` rather than at most `bits`, and set the low bit because every even
/// number above 2 is composite. Candidates are therefore uniform over the odd
/// integers in `[2^(bits−1), 2^bits)`. The single prime that forcing excludes
/// is 2 itself, so `bits == 2` always yields 3.
///
/// The screen is [`crate::is_probable_prime`], Miller-Rabin against the fixed
/// twelve-prime base set: a proof of primality below `3.317 × 10^24` and a
/// strong probable-prime test above it. A fixed base set is the right choice
/// precisely here, where the candidate is drawn by this function rather than
/// supplied by an adversary who could target the bases; the name is exact, and
/// a caller wanting more assurance at large widths adds witnesses of its own
/// through [`crate::miller_rabin_witness`].
///
/// Termination is probabilistic, as the module note explains: the density of
/// primes makes the expected number of rounds finite under a generator with
/// full-range output, but a generator that returns the same bytes every call
/// will retry a fixed composite forever.
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

    /// splitmix64 — Steele, Lea & Flood, *Fast Splittable Pseudorandom Number
    /// Generators*, OOPSLA 2014: a 64-bit additive counter through a fixed
    /// finalizing mix. Reproducible from a seed and adequate for exercising
    /// the samplers; it is not a CSPRNG and is confined to these tests.
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
        // Coprime-to-0 is degenerate (only 1 qualifies): None, not an
        // unbounded rejection loop (review §4.10).
        assert_eq!(
            random_coprime_below(&mut rng, &bound, &BigUint::zero()),
            None
        );
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
