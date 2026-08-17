//! Random sampling of integers and probable primes.
//!
//! rump chooses no entropy source: every function is driven by a
//! caller-supplied [`RandomSource`], and the output is exactly as good as that
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
//! the supplied generator, not a guarantee of the code. Each sampler carries
//! a stall guard that panics rather than looping forever on the degenerate
//! sources it can soundly detect — the posture
//! `PolyMod::equal_degree_split` takes — and every guard is sized so a
//! working generator trips it with probability at most `e⁻¹¹¹ ≈ 2⁻¹⁶⁰`
//! (the loosest of the individual bounds; each function documents its
//! own), so the panic is a diagnosis, never a sampling accident. What
//! each guard can detect differs, and the one gap is stated below rather
//! than papered over:
//!
//! - [`random_below`] and [`random_nonzero_below`] accept every draw with
//!   probability at least one half regardless of arguments, so they bound
//!   consecutive *rejections* and catch every degenerate source.
//! - [`random_probable_prime`]'s acceptance density is a function of the
//!   width alone, so its rejection bound scales with the width (`64·bits`)
//!   and likewise catches every degenerate source; a *constant* generator
//!   additionally fails fast, via a repeat detector that skips the
//!   Miller–Rabin re-screen.
//! - [`random_coprime_below`] is the one sampler where no *usable*
//!   rejection count exists. A primorial modulus legitimately leaves 1 as
//!   the only unit below the bound, so acceptance can be as thin as
//!   `1/(upper − 1)`; the only sound count therefore scales with the
//!   bound itself (Θ(upper) draws), which is unreachable for a
//!   multiprecision bound. Its guard instead detects a *pinned* generator
//!   (the same rejected candidate over and over); a degenerate source
//!   that cycles among several rejected values is statistically
//!   indistinguishable from a legitimately unlucky run against a sparse
//!   unit set, does not trip the guard, and remains the caller's to
//!   avoid.
//!
//! Degenerate *arguments* are still reported as `None`.

use crate::bigint::BigUint;
use crate::number_theory::{gcd, is_probable_prime};

/// Consecutive fruitless draws after which [`random_below`] and
/// [`random_nonzero_below`] conclude the generator is broken. Both loops
/// accept each draw with probability at least one half, so a working
/// generator survives this bound with probability at most `2⁻²⁵⁶` —
/// the same constant, and the same reasoning, as
/// `PolyMod::equal_degree_split`'s stall guard.
const MAX_REJECTED_DRAWS: usize = 256;

/// Consecutive draws of the *same rejected candidate* after which a
/// sampler declares the generator pinned — the signature no working
/// generator produces. Whenever a rejection is possible at all the
/// candidate space holds at least two values, so independent uniform
/// draws repeat the previous one with probability at most one half, and
/// 256 consecutive repeats has probability at most `2⁻²⁵⁶` — the same
/// constant, and the same reasoning, as the stall guard in
/// `PolyMod::equal_degree_split`.
///
/// Two samplers use it, in different roles. For [`random_coprime_below`]
/// it is the *only* guard, because no usable rejection count exists there
/// (see the module note). For [`random_probable_prime`] it is the fast
/// *inner* guard beneath the width-scaled rejection count: it exists so a
/// constant generator fails after one Miller–Rabin screen rather than
/// after `64·bits` of them.
const MAX_IDENTICAL_REJECTIONS: usize = 256;

/// A source of random bytes.
///
/// The one-method contract every sampler here needs; implement it by filling
/// `dest` completely. No quality is implied — supplying a cryptographically
/// strong generator is the caller's responsibility when the use demands one.
/// The samplers assume only that successive fills are independent draws;
/// nothing in this module inspects, reseeds, or forks the generator.
pub trait RandomSource {
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
/// rem `upper_exclusive` would over-represent the low residues whenever
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
///
/// # Panics
///
/// Panics after 256 consecutive rejections (`MAX_REJECTED_DRAWS`). Each draw is
/// accepted with probability at least one half, so for a working generator
/// this has probability at most `2⁻²⁵⁶`; it fires only on a generator whose
/// output is confined to the rejected region (see the module note).
#[must_use]
pub fn random_below<R: RandomSource + ?Sized>(
    rng: &mut R,
    upper_exclusive: &BigUint,
) -> Option<BigUint> {
    if upper_exclusive.is_zero() {
        return None;
    }

    let bits = upper_exclusive.bits();
    let mut bytes = vec![0u8; bits.div_ceil(8)];
    let excess_bits = bytes.len() * 8 - bits;
    let top_mask = 0xff_u8 >> excess_bits;

    for _ in 0..MAX_REJECTED_DRAWS {
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
    panic!(
        "random_below rejected {MAX_REJECTED_DRAWS} consecutive draws \
         (each is accepted with probability at least 1/2): \
         the supplied RandomSource yields no usable entropy"
    );
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
///
/// # Panics
///
/// Panics after 256 consecutive zero draws (`MAX_REJECTED_DRAWS`; probability
/// at most `2⁻²⁵⁶` for a working generator, since zero is at most one
/// candidate in two), or as [`random_below`] does if the generator stalls
/// below it.
#[must_use]
pub fn random_nonzero_below<R: RandomSource + ?Sized>(
    rng: &mut R,
    upper_exclusive: &BigUint,
) -> Option<BigUint> {
    if upper_exclusive <= &BigUint::one() {
        return None;
    }

    for _ in 0..MAX_REJECTED_DRAWS {
        let candidate = random_below(rng, upper_exclusive)?;
        if !candidate.is_zero() {
            return Some(candidate);
        }
    }
    panic!(
        "random_nonzero_below drew zero {MAX_REJECTED_DRAWS} times in a row \
         (zero is at most one candidate in two): \
         the supplied RandomSource yields no usable entropy"
    );
}

/// Draw a random integer in `[1, upper_exclusive)` that is coprime to
/// `coprime_to`, uniformly over that set.
///
/// This is the shape needed to draw a fresh random unit rem `n`: pass
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
///
/// # Panics
///
/// Panics when the same rejected candidate is drawn 256 times in a row
/// (`MAX_IDENTICAL_REJECTIONS` — see that constant for why only a pinned
/// generator can do this; a mere run of distinct non-coprime draws never
/// trips it, however long, because the acceptance density is the
/// arguments' business), or as [`random_below`] does if the generator
/// stalls below it. This guard covers only the pinned case: a degenerate
/// generator cycling among several non-coprime values hangs this function,
/// as the module note explains — no argument-independent bound can
/// distinguish it from an unlucky legitimate run.
#[must_use]
pub fn random_coprime_below<R: RandomSource + ?Sized>(
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
    // Stall detection watches for a *pinned* generator (the previous
    // rejected candidate drawn again), not for a long run of rejections —
    // the latter can be legitimate when the arguments make units scarce.
    let mut last_rejected: Option<BigUint> = None;
    let mut stalled = 0usize;
    loop {
        let candidate = random_nonzero_below(rng, upper_exclusive)?;
        if gcd(&candidate, coprime_to) == BigUint::one() {
            return Some(candidate);
        }
        if last_rejected.as_ref() == Some(&candidate) {
            stalled += 1;
            assert!(
                stalled < MAX_IDENTICAL_REJECTIONS,
                "random_coprime_below drew the same rejected candidate \
                 {MAX_IDENTICAL_REJECTIONS} times in a row: \
                 the supplied RandomSource yields no usable entropy"
            );
        } else {
            stalled = 0;
            last_rejected = Some(candidate);
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
/// twelve-prime base set: a proof of primality below `ψ₁₂ ≈ 3.19 × 10^23` and a
/// strong probable-prime test above it. A fixed base set is the right choice
/// precisely here, where the candidate is drawn by this function rather than
/// supplied by an adversary who could target the bases; the name is exact, and
/// a caller wanting more assurance at large widths adds witnesses of its own
/// through [`crate::miller_rabin_witness`].
///
/// Termination under a working generator is probabilistic, and the stall
/// guard is two-layered because the failure modes differ in cost. By the
/// prime number theorem the density of primes among the odd `bits`-bit
/// candidates is about `2/(bits·ln 2)`, so the expected number of rounds
/// is about `0.35·bits` under a generator with full-range output. The
/// outer guard bounds fruitless rounds at `64·bits` — the cap scales with
/// the width, so a working generator survives it with probability about
/// `(1 − 2/(bits·ln 2))^(64·bits) ≈ e⁻¹⁸⁵` by the asymptotic density, and
/// below `e⁻¹¹¹` unconditionally at every width in range — and catches
/// every degenerate source, including one cycling among several
/// composites. The unconditional bound splits by width: for `bits ≥ 6`,
/// `π(2x) − π(x) > (3/5)·x/ln x` (Rosser and Schoenfeld 1962; on the
/// citation-check list, the inequality itself verified numerically to
/// `10⁷`) bounds the density below by `1.2/((bits−1)·ln 2)`, giving
/// survival under `e⁻¹¹¹`; the four smaller widths check exhaustively —
/// 2 and 3 contain no composite at all, and the worst case is `bits = 4`
/// (density 1/2 over 256 rounds, survival `2⁻²⁵⁶ = e⁻¹⁷⁷`). The inner guard makes the common
/// broken source fail fast: a candidate equal to the previous rejected
/// one is already known composite, so it skips the Miller–Rabin re-screen
/// and trips its own 256-repeat assertion — one screen and 256
/// comparisons for a constant generator, not hours of full-width
/// exponentiation at large widths. (At `bits ≤ 4` the outer cap of
/// `64·bits ≤ 256` rounds expires first, so a pinned generator there
/// panics with the round-count message instead; the cost is the same.)
///
/// # Panics
///
/// Panics when the same rejected candidate is drawn 256 times in a row
/// (`MAX_IDENTICAL_REJECTIONS`; a working generator repeats the previous
/// draw with probability at most 1/4 per round whenever any composite is
/// in range, so this has probability at most `4⁻²⁵⁵`), or after `64·bits`
/// fruitless rounds in total (probability below `e⁻¹¹¹` for a working
/// generator at every width, by the split argument above).
#[must_use]
pub fn random_probable_prime<R: RandomSource + ?Sized>(
    rng: &mut R,
    bits: usize,
) -> Option<BigUint> {
    if bits < 2 {
        return None;
    }

    let mut bytes = vec![0u8; bits.div_ceil(8)];
    let top_bit = (bits - 1) % 8;
    let excess_bits = bytes.len() * 8 - bits;
    let top_mask = 0xff_u8 >> excess_bits;
    let mut last_rejected: Option<BigUint> = None;
    let mut stalled = 0usize;
    for _ in 0..64 * bits {
        rng.fill_bytes(&mut bytes);
        bytes[0] &= top_mask;
        // Force the requested bit length by setting the top significant bit,
        // then force oddness because every even candidate above 2 is composite.
        bytes[0] |= 1u8 << top_bit;
        let last = bytes.len() - 1;
        bytes[last] |= 1;

        let candidate = BigUint::from_be_bytes(&bytes);
        crate::scrub::zeroize_slice(bytes.as_mut_slice());
        if last_rejected.as_ref() == Some(&candidate) {
            // Known composite from the previous round: count the stall and
            // skip the screen it already failed.
            stalled += 1;
            assert!(
                stalled < MAX_IDENTICAL_REJECTIONS,
                "random_probable_prime drew the same composite \
                 {MAX_IDENTICAL_REJECTIONS} times in a row at {bits} bits: \
                 the supplied RandomSource yields no usable entropy"
            );
            continue;
        }
        if is_probable_prime(&candidate) {
            return Some(candidate);
        }
        stalled = 0;
        last_rejected = Some(candidate);
    }
    panic!(
        "random_probable_prime found no prime in {} rounds at {bits} bits \
         (about 185 times the expected search length): \
         the supplied RandomSource yields no usable entropy",
        64 * bits
    );
}

#[cfg(test)]
mod tests {
    use super::{
        random_below, random_coprime_below, random_nonzero_below, random_probable_prime,
        RandomSource,
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

    impl RandomSource for SplitMix64 {
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

    impl RandomSource for ZeroRng {
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
            bound = bound.sub(&BigUint::one());
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

    /// A generator pinned to one byte value — the degenerate source the
    /// stall guards exist to diagnose.
    struct ConstRng(u8);

    impl RandomSource for ConstRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(self.0);
        }
    }

    #[test]
    #[should_panic(expected = "the supplied RandomSource yields no usable entropy")]
    fn stalled_generator_fails_loudly_in_random_below() {
        // Bound 5 has bit length 3, so the masked draw is always 0x07 = 7,
        // which is rejected forever: the guard must fire, not spin.
        let mut rng = ConstRng(0xff);
        let _ = random_below(&mut rng, &BigUint::from_u64(5));
    }

    #[test]
    #[should_panic(expected = "the supplied RandomSource yields no usable entropy")]
    fn stalled_generator_fails_loudly_in_random_nonzero_below() {
        // Every draw is zero — always in range, never non-zero.
        let mut rng = ZeroRng;
        let _ = random_nonzero_below(&mut rng, &BigUint::from_u64(5));
    }

    #[test]
    #[should_panic(expected = "the supplied RandomSource yields no usable entropy")]
    fn stalled_generator_fails_loudly_in_random_coprime_below() {
        // Every draw is 2 — in range, non-zero, and never coprime to 6.
        let mut rng = ConstRng(0x02);
        let _ = random_coprime_below(&mut rng, &BigUint::from_u64(3), &BigUint::from_u64(6));
    }

    #[test]
    #[should_panic(expected = "the supplied RandomSource yields no usable entropy")]
    fn stalled_generator_fails_loudly_in_random_probable_prime() {
        // Forcing the top and low bits of an all-zero draw pins the
        // candidate at 2^7 + 1 = 129 = 3·43, composite forever. This trips
        // the fast inner guard (repeat detection) after 256 comparisons.
        let mut rng = ZeroRng;
        let _ = random_probable_prime(&mut rng, 8);
    }

    /// A generator cycling between two byte values — degenerate but not
    /// pinned, so only the outer fruitless-round bound can catch it.
    struct TwoCycleRng {
        first: u8,
        second: u8,
        flip: bool,
    }

    impl RandomSource for TwoCycleRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(if self.flip { self.second } else { self.first });
            self.flip = !self.flip;
        }
    }

    #[test]
    #[should_panic(expected = "the supplied RandomSource yields no usable entropy")]
    fn cycling_generator_fails_loudly_in_random_probable_prime() {
        // Bytes 0x00 and 0x04 force the 8-bit candidates 129 = 3·43 and
        // 133 = 7·19, alternating: never the same as the previous draw, so
        // the repeat detector stays quiet and the 64·bits fruitless-round
        // bound must fire instead.
        let mut rng = TwoCycleRng {
            first: 0x00,
            second: 0x04,
            flip: false,
        };
        let _ = random_probable_prime(&mut rng, 8);
    }
}
