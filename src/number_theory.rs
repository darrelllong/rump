//! Deterministic number theory over [`BigUint`].
//!
//! Everything here is a pure function of its inputs: gcd and lcm by Euclid,
//! the Jacobi symbol by binary reciprocity, modular exponentiation and
//! inversion, and fixed-base Miller-Rabin. Randomized prime *generation* and
//! adversarially hardened primality testing live with their consumers (the
//! parent cryptography crate), where the entropy source and hash live.

use crate::bigint::{BigInt, BigUint, MontgomeryCtx};

/// Fixed Miller-Rabin witness set used by the bigint probable-prime test.
///
/// These twelve small prime bases give a deterministic, repeatable witness
/// schedule. They are the classic "small prime" bases through `37`.
///
/// Notes on determinism:
/// - For all odd `n < 2^64`, published smaller witness sets are already
///   deterministic; this superset is therefore deterministic in that range.
/// - For larger `BigUint` candidates this remains a strong fixed-basis
///   probable-prime test, but not a proof of primality.
const MR_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Trial-division sieve primes checked before the Miller-Rabin stage.
///
/// Cheap remainders here discard most composites before the code pays for any
/// modular exponentiation.
const SMALL_TRIAL_PRIMES: [u16; 168] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
];

/// Greatest common divisor by Euclid's algorithm.
#[must_use]
pub fn gcd(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    let mut current = lhs.clone();
    let mut next = rhs.clone();
    while !next.is_zero() {
        let remainder = current.modulo(&next);
        current = next;
        next = remainder;
    }
    current
}

/// Least common multiple.
///
/// This is the Carmichael-function building block used by the RSA code: the
/// Python reference chooses `lambda = lcm(p - 1, q - 1)` rather than Euler's
/// totient because the private exponent only needs to invert modulo the
/// exponent cycle length.
#[must_use]
pub fn lcm(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    if lhs.is_zero() || rhs.is_zero() {
        return BigUint::zero();
    }

    let divisor = gcd(lhs, rhs);
    let (quotient, remainder) = lhs.div_rem(&divisor);
    debug_assert!(remainder.is_zero(), "gcd divides the left operand exactly");
    quotient.mul_ref(rhs)
}

/// Jacobi symbol `(a/n)` for odd `n`, or `None` when `n` is even or zero.
///
/// Binary algorithm via quadratic reciprocity (Menezes, van Oorschot &
/// Vanstone, *Handbook of Applied Cryptography*, Algorithm 2.149): strip
/// factors of two using the supplement `(2/n) = (-1)^((n^2 - 1)/8)` — a sign
/// flip exactly when `n ≡ 3, 5 (mod 8)` — then swap the arguments, paying the
/// reciprocity sign flip when both are `≡ 3 (mod 4)`, and reduce. Runs in
/// `O(log a · log n)` bit operations like the gcd it shadows.
///
/// For prime `n` this is the Legendre symbol: `1` for quadratic residues,
/// `-1` for non-residues, `0` when `n` divides `a`. `(a/1) = 1` by the
/// empty-product convention.
#[must_use]
pub fn jacobi(a: &BigUint, n: &BigUint) -> Option<i8> {
    if n.is_zero() || !n.is_odd() {
        return None;
    }

    let mut a = a.modulo(n);
    let mut n = n.clone();
    let mut sign = 1i8;

    while !a.is_zero() {
        // Strip a's factors of two; each one contributes the (2/n) supplement.
        let mut twos = 0usize;
        while !a.bit(twos) {
            twos += 1;
        }
        if twos % 2 == 1 {
            let n_mod_8 = n.rem_u64(8);
            if n_mod_8 == 3 || n_mod_8 == 5 {
                sign = -sign;
            }
        }
        a.shr_bits(twos);

        // Reciprocity: with both arguments now odd, (a/n) and (n/a) agree
        // unless both are ≡ 3 (mod 4).
        if a.rem_u64(4) == 3 && n.rem_u64(4) == 3 {
            sign = -sign;
        }
        core::mem::swap(&mut a, &mut n);
        a = a.modulo(&n);
    }

    // The loop preserves (a/n) up to the accumulated sign; it ends with the
    // gcd in `n`. A gcd above one means a and the original n share a factor,
    // where the symbol is zero by definition.
    if n.is_one() {
        Some(sign)
    } else {
        Some(0)
    }
}

/// `base^exponent mod modulus` by repeated squaring.
///
/// # Panics
///
/// Panics if `modulus == 0`.
#[must_use]
pub fn mod_pow(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
    assert!(!modulus.is_zero(), "modulus must be non-zero");
    if modulus == &BigUint::one() {
        return BigUint::zero();
    }
    if let Some(ctx) = MontgomeryCtx::new(modulus) {
        return ctx.pow(base, exponent);
    }

    let mut result = BigUint::one();
    let mut power = base.modulo(modulus);
    for bit in 0..exponent.bits() {
        if exponent.bit(bit) {
            result = BigUint::mod_mul(&result, &power, modulus);
        }
        power = BigUint::mod_mul(&power, &power, modulus);
    }
    result
}

fn decompose_n_minus_one(n: &BigUint) -> (BigUint, usize) {
    let mut odd_factor = n.sub_ref(&BigUint::one());
    let mut two_adic_exponent = 0usize;
    // Write `n - 1 = d * 2^s` with `d` odd. Miller-Rabin uses `d` as the
    // first exponent and then squares through the `2^s` chain.
    while !odd_factor.is_odd() {
        odd_factor.shr1();
        two_adic_exponent += 1;
    }
    (odd_factor, two_adic_exponent)
}

fn is_witness(
    base: &BigUint,
    ctx: &MontgomeryCtx,
    odd_factor: &BigUint,
    two_adic_exponent: usize,
) -> bool {
    let one = BigUint::one();
    let n_minus_one = ctx.modulus().sub_ref(&one);
    let mut value = ctx.pow(base, odd_factor);

    // Miller-Rabin witness test: a non-trivial square root of 1 proves
    // compositeness, and failing to end at 1 is the usual Fermat backstop.
    for _ in 0..two_adic_exponent {
        let next = ctx.square(&value);
        if next == one && value != one && value != n_minus_one {
            return true;
        }
        value = next;
    }

    value != one
}

/// Miller-Rabin probable-prime test with a fixed witness set.
///
/// Appropriate for the caller's own randomly generated candidates, where a
/// fixed base set already rejects composites with overwhelming probability.
/// For values that arrive from an untrusted source (parsed keys,
/// peer-supplied domain parameters) a fixed base set can be defeated by a
/// purpose-built strong pseudoprime; harden with additional
/// candidate-derived witnesses via [`miller_rabin_witness`], as the parent
/// cryptography crate's `is_probable_prime_untrusted` does.
#[must_use]
pub fn is_probable_prime(n: &BigUint) -> bool {
    mr_probable_prime(n, &MR_BASES)
}

/// Miller-Rabin using explicit witness bases.
#[must_use]
pub fn is_probable_prime_with_bases(candidate: &BigUint, bases: &[u64]) -> bool {
    mr_probable_prime(candidate, bases)
}

/// Run one Miller-Rabin round: does `witness` prove `candidate` composite?
///
/// The reusable primitive behind the probable-prime tests, public so callers
/// can bring their own witness schedule (for example, witnesses derived by
/// hashing an untrusted candidate). Returns `false` — no compositeness
/// proven — for witnesses congruent to `0`, `±1 (mod candidate)`, which can
/// never testify, and treats even or trivial candidates as composite.
#[must_use]
pub fn miller_rabin_witness(candidate: &BigUint, witness: &BigUint) -> bool {
    let Some(ctx) = MontgomeryCtx::new(candidate) else {
        // Even or zero candidates: composite (or degenerate) by inspection.
        return true;
    };
    if candidate.is_one() {
        return true;
    }

    let n_minus_one = candidate.sub_ref(&BigUint::one());
    let witness = witness.modulo(candidate);
    if witness <= BigUint::one() || witness == n_minus_one {
        return false;
    }

    let (odd_factor, two_adic_exponent) = decompose_n_minus_one(candidate);
    is_witness(&witness, &ctx, &odd_factor, two_adic_exponent)
}

fn mr_probable_prime(candidate: &BigUint, bases: &[u64]) -> bool {
    if candidate.is_zero() {
        return false;
    }
    if candidate == &BigUint::one() {
        return false;
    }

    for &prime in &SMALL_TRIAL_PRIMES {
        let prime = u64::from(prime);
        let remainder = candidate.rem_u64(prime);
        if remainder == 0 {
            // A small prime divides itself as well as its composite
            // multiples. For candidates below 2^10, the residue modulo 2^10
            // distinguishes the identity case without allocating a temporary
            // BigUint for every sieve entry.
            if candidate.bits() <= 10 && candidate.rem_u64(1u64 << 10) == prime {
                return true;
            }
            return false;
        }
    }

    if bases.is_empty() {
        return false;
    }

    let Some(ctx) = MontgomeryCtx::new(candidate) else {
        return false;
    };
    let n_minus_one = candidate.sub_ref(&BigUint::one());
    let (odd_factor, two_adic_exponent) = decompose_n_minus_one(candidate);

    for &base in bases {
        let witness = BigUint::from_u64(base);
        // Bases `>= n - 1` add no information here: `n - 1` is the trivial
        // `-1 mod n` case, and larger bases reduce to residues that a smaller
        // representative would already cover.
        if witness >= n_minus_one {
            continue;
        }
        if is_witness(&witness, &ctx, &odd_factor, two_adic_exponent) {
            return false;
        }
    }

    true
}

/// Multiplicative inverse `a^{-1} mod n`, if it exists.
#[must_use]
pub fn mod_inverse(a: &BigUint, n: &BigUint) -> Option<BigUint> {
    if n.is_zero() {
        return None;
    }

    let mut t = BigInt::zero();
    let mut new_t = BigInt::from_biguint(BigUint::one());
    let mut r = n.clone();
    let mut new_r = a.modulo(n);

    // Extended Euclid. If the gcd ends at 1, the tracked coefficient `t`
    // satisfies `t * a ≡ 1 (mod n)` and is therefore the modular inverse.
    while !new_r.is_zero() {
        let (quotient, remainder) = r.div_rem(&new_r);
        let next_t = t.sub_ref(&new_t.mul_biguint_ref(&quotient));
        t = new_t;
        new_t = next_t;
        r = new_r;
        new_r = remainder;
    }

    if !r.is_one() {
        return None;
    }

    Some(t.modulo_positive(n))
}

#[cfg(test)]
mod tests {
    use super::{
        gcd, is_probable_prime, is_probable_prime_with_bases, jacobi, lcm, miller_rabin_witness,
        mod_inverse, mod_pow,
    };
    use crate::bigint::BigUint;

    /// splitmix64 (Steele, Lea & Flood 2014): deterministic scattered draws
    /// with no dependency; not a CSPRNG and not meant to be one.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Draw a value in `[1, upper)` by rejection from random 64-bit words.
    fn draw_below(rng: &mut SplitMix64, upper: &BigUint) -> BigUint {
        let words = upper.bits().div_ceil(64);
        loop {
            let mut bytes = Vec::with_capacity(words * 8);
            for _ in 0..words {
                bytes.extend_from_slice(&rng.next_u64().to_be_bytes());
            }
            let candidate = BigUint::from_be_bytes(&bytes).modulo(upper);
            if !candidate.is_zero() {
                return candidate;
            }
        }
    }

    fn mersenne(exponent: usize) -> BigUint {
        let mut value = BigUint::zero();
        value.set_bit(exponent);
        value.sub_ref(&BigUint::one())
    }

    fn biguint_from_hex(hex: &str) -> BigUint {
        let padded = if hex.len() % 2 == 1 {
            format!("0{hex}")
        } else {
            hex.to_string()
        };
        let bytes: Vec<u8> = (0..padded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&padded[i..i + 2], 16).expect("hex vector"))
            .collect();
        BigUint::from_be_bytes(&bytes)
    }

    /// Reference vectors computed by GMP 6.3.0's mpz_jacobi (see
    /// scripts/bench_gmp.sh for the toolchain): an independent oracle for the
    /// binary reciprocity algorithm. Triples are (a, n, (a/n)) with a and n in
    /// hex, spanning 8- to 1024-bit odd moduli, a below/at/above n, shared
    /// factors, and the (2/n) supplement cases.
    const GMP_JACOBI_VECTORS: &[(&str, &str, i8)] = &[
        ("8a", "bf", 1),
        ("f6ad", "bf", -1),
        ("be", "bf", -1),
        ("c0", "c1", 1),
        ("7e10", "c1", -1),
        ("c0", "c1", 1),
        ("53", "a9", 1),
        ("18a3", "a9", 1),
        ("a8", "a9", 1),
        ("32", "fb", -1),
        ("7110", "fb", 1),
        ("fa", "fb", -1),
        ("36", "db", 0),
        ("1bc9", "db", 0),
        ("da", "db", -1),
        ("a5e2", "8d27", 1),
        ("509c55fd", "8d27", -1),
        ("8d26", "8d27", -1),
        ("fd96", "9003", -1),
        ("19e41c1a", "9003", 1),
        ("9002", "9003", -1),
        ("8e6f", "ffbd", 1),
        ("7cfdf0cd", "ffbd", 0),
        ("ffbc", "ffbd", 1),
        ("229a", "a381", 1),
        ("66bcc7a2", "a381", -1),
        ("a380", "a381", 1),
        ("c30a", "a0c7", -1),
        ("e16d0717", "a0c7", 1),
        ("a0c6", "a0c7", -1),
        ("6d873e188a3b18af", "98976396c66cf253", -1),
        ("b81585e184058393b4e70630249faa41", "98976396c66cf253", 1),
        ("98976396c66cf252", "98976396c66cf253", -1),
        ("ec55142e656f893f", "e031880a1a92cd89", -1),
        ("d93ee354868a5658d359a944ff748928", "e031880a1a92cd89", 0),
        ("e031880a1a92cd88", "e031880a1a92cd89", 1),
        ("29ffabdbcb4a054", "d120bc92095cbe79", 0),
        ("c8122140efaf5fc236c6ce77b0b2bf68", "d120bc92095cbe79", 0),
        ("d120bc92095cbe78", "d120bc92095cbe79", 1),
        ("1365749bb30353c7", "a664b4d6936fd78f", 1),
        ("54e5f8197da36f2d88e051f28b6bf16c", "a664b4d6936fd78f", -1),
        ("a664b4d6936fd78e", "a664b4d6936fd78f", -1),
        ("d9c52e47915040ed", "b35a54c1cb764e05", 1),
        ("8802aa2cd2117e21c785a6949cab78e3", "b35a54c1cb764e05", -1),
        ("b35a54c1cb764e04", "b35a54c1cb764e05", 1),
        ("152a452713c13cf28", "11d0822f11e0a04f9", 0),
        ("2e39313c5e003a23ddb0ec24261cdea75", "11d0822f11e0a04f9", -1),
        ("11d0822f11e0a04f8", "11d0822f11e0a04f9", 1),
        ("10c82f8b0beeac785", "186a236eddb59c71d", -1),
        ("275d1f9e5e55cd14860166699856c1cb2", "186a236eddb59c71d", -1),
        ("186a236eddb59c71c", "186a236eddb59c71d", 1),
        ("1c86648969034aad7", "1cdc1ca9241c8c89f", 1),
        ("1cb550f0a9ec2f62bb89ce5ddc855a5b5", "1cdc1ca9241c8c89f", 0),
        ("1cdc1ca9241c8c89e", "1cdc1ca9241c8c89f", -1),
        ("10058c5f107e77202", "1b9fe84737e9ce40d", 1),
        ("3f312533cefb5bf51d382b7d7cd061dc0", "1b9fe84737e9ce40d", 0),
        ("1b9fe84737e9ce40c", "1b9fe84737e9ce40d", 1),
        ("1921ff6d416fb942c", "1c1f064b7a273494d", -1),
        ("1185aff0ecc65131f898a25cc3f5ee120", "1c1f064b7a273494d", -1),
        ("1c1f064b7a273494c", "1c1f064b7a273494d", 1),
        ("a9438fc24555f3ba25098f47a0c9900c", "9f7d286288234afda8fb40e225691723", -1),
        ("e4ec421914dfe4a5640b28bb6881fc7f8e289c18cfd6249db02d6883e765c44b", "9f7d286288234afda8fb40e225691723", -1),
        ("9f7d286288234afda8fb40e225691722", "9f7d286288234afda8fb40e225691723", -1),
        ("2e9c936bbaf896c50b5e6b167b1b9b4e", "aa61f09956d095dbd4b7468ec84e0f09", -1),
        ("986fceb34ef71509b8273d4449310beb4ec320b5c8da56fa93d62dec246e4fe3", "aa61f09956d095dbd4b7468ec84e0f09", -1),
        ("aa61f09956d095dbd4b7468ec84e0f08", "aa61f09956d095dbd4b7468ec84e0f09", 1),
        ("39f0134ade9c857546025fae1e69df0", "df4ea3ac476c5572a46d159b2243e3c3", -1),
        ("b1d0277f524cbac76ced302f001d147380c8c4742b4953bcc127113da2a5654d", "df4ea3ac476c5572a46d159b2243e3c3", 0),
        ("df4ea3ac476c5572a46d159b2243e3c2", "df4ea3ac476c5572a46d159b2243e3c3", -1),
        ("1d73044ce3c70a32aa25c14a375ba74d", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("7f5a737dcdc5d2e0bd0fff8160b765080bb3bc5ba396eae348089aee98c35a68", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("eefd0a7d626f5ad5293715d4703ae73a", "eefd0a7d626f5ad5293715d4703ae73b", -1),
        ("740b15bc6c4d79966d8a27bfd52bedf", "91ba073343b37c454b8cf8cf7e4f35c9", -1),
        ("a6aae2ae5a925cd4cc66688c84e300324fed8cf4c9206b1dbd5f674a3e070297", "91ba073343b37c454b8cf8cf7e4f35c9", -1),
        ("91ba073343b37c454b8cf8cf7e4f35c8", "91ba073343b37c454b8cf8cf7e4f35c9", 1),
        ("52aaff89c15602f4a88ea950b46abf84e54522d39ce8f27eec3974977e61ba8d", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("88a12277a1a579d30a67031c3caacf9cdf22ab6582f343320f46902a22bc4cb7376ad797f98f284733a8ccb24fd8e70550e85c2ad3ec06d8fd133637f2b93e65", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0da", "923b5aa7d3b160ff8e18d442cc210d0ba1ec297bc3692111abc437e3e7b3e0db", -1),
        ("789a5350f58ab019a1c9e51a4b0eefe608fe1ce5ce9099bb9e571491251427f1", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", 0),
        ("c269e9c35bd61340219ab4a1c6b173ae584f51f0756588eb5762a39b9a80d599afea93e3d209dbe35d27154ff743bc0f28bf4f564c6ebce34ddc8b6d56f3993f", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", -1),
        ("faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc4", "faa7fc7723c3c19c7295126de4aa0f813076731e0d96990475a63529ee19dcc5", 1),
        ("97ef3e2f6910b313fb1de73398a332b2157d9dae447d79cf85d58bf099d5e9ec", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 0),
        ("6f9f85fc79d6c631cb2dcf18c74eea09975d332f89a056208ae53efb17ee8f7455e28557ed30196c46977bd65e18a359c873750472125974db5d33b537727c41", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 0),
        ("a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c8", "a94549240d2d4ed2779f54ebd3ea0f915920f807abc2b00ab0a4473b9c40e9c9", 1),
        ("c4152c3771950fe29dc8f7459471fed21d6f9c15b94d5211b96940d71e2689b0", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", -1),
        ("d094982cf17c83358ae3114b0bbc7c7b9bca1e85708b0a5fbd92479844ca562158abdd0c93f3b0028eeb436021bc064093b881877c366831e827f2b59bf34a4a", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", 1),
        ("88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02afe", "88add2d7cba103dc87c7498dbd8d37e947a0644a907c123f22a2bdcccff02aff", -1),
        ("933086656e1af289d42f23adbe850f8495b758176826a20b6cd87f29ace65c91", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("9ce2a2929ae038d6ebbd86048cc8fb2599e90ee288a040385fcd6523fd3bdc210572431aba1611d9a991d585e819a4c94f109e64cb8dae5ede9e2acd7d83332d", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891a", "a8cb19945ea56c44e14585fe276fa8cef4c2bc27ffa33d1d290a1141783e891b", -1),
        ("17d4de0e6921e6da75d313b33c8b03e79df71f7e99e271115b2d299aa6c5e9953b50321429f39ff48a7abd8489e1781306bebd684276fa1c44319d481d8d70f1688", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("1937a228664f6e2560b0c7b2628854d08ed07663dba0ab47264c70d6c39bf3f391dbdab37db4eb3fb46a7b05399884d9ceb5040a641d83bf7a4785d103bf60de2f46f6bb01520940332ebc227184127d34ff745e273cb30ea6c35bcd24447c41f8c23254891531e7f42ed3a22589cf49a9eb53d283a19149b65d70d51cf256e172fd7", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cc", "1a848bc61fb55346c87fc5a6e01f3ff1a39545f9fe5cfa9f4f2418cb3bceacd9c5e5330e76cf0a1fff115ea3316c538bfcb5a7811f7706ddac73ff72219ed8324cd", 1),
        ("a29602617c33a9bca435546f7a1b475e691a63f90caffa41a39a9f2b2ed54b535ad1ba14f8eb9075def2932e9b9a36b1b7f5baff5bf3e130c564dce07bbc1de961", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", 1),
        ("85179c2362148bfb3ce9b0be61475461080349e33696ec7ebf7ad9637b563f7013c58898632a3366abd6f82ba5e26dd512c360942a5943deff8786b56caed03052c1ef55e35ad5d84c7b6dfaf237a68482fc7817ed98234712413f6ad271a349cfa6172b931a66a65ced240f3d87294a26c2e3b78bda9c7e4e730ba83b0f0752e6fc", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", 1),
        ("1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394fe", "1c7884db9121df2d851c507756c187fab4b1131ee1ee188d8f38a08e6e057654fd0eba10673c0a72d41d8111c2e5d242e9f3db9451b45667a08f60e3e8aa63394ff", -1),
        ("b27d4208b8f77f7e1a61b13ee0579a5caf01ff8023bd92369885b0e97d44eb9c21ccfc6247f712b2b8efb191a832730770c5e9d74c7c4ec1d064decb334984c78c", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", 1),
        ("2fcdec8864cf66f0e83fbcc94ac19e99a2ed4cbb9073aeefbe0e2f538a804e611b9e7b61781727a7ec1eef061026e5dd0b985e12f65c0845f4a45db18fc231931d8c17c432418205d23506dc0ab70edc27d0175b797f3b567551e9d6167c88ee221e8263c579488ec45707b2ab4f472aba0ad4f0c6c15afe525920a86b41e2fcf3114", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", 1),
        ("10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524cea", "10bada040e275c63cb510bb620058ff52cd9c0b1984b293de8b27267d017e0118549980c5a14e570f9d2bae091a3ae32101e8890502ba6c762ac052efae0e524ceb", -1),
        ("81aa611e687bb0805671a0343c0a63d848d96785f5cddf4200b31c897a8e35d60a25275bf05232e71d6bc90a12532a042cf3d0d10c6ba3ddaf0fa6726953e27aea", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", -1),
        ("68c2528e9b792684f96da6d0a7a4e3fd5bedaf6241708679189699e8d40f378855f2710d136aa4deb549981b3c2e62fdc93d3fd7089ba5391ae47c8ad025707973fa692f1c1344fb9564547a13c3a56bf3f1db749ff7405124fb2b5903617ec4b8509659882be92d9e0ce4ce91468b611c177272471ab317c4d1d7c179dc13495d2c", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", -1),
        ("1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b4", "1bd800115259fdd8d5f446716cd292647332e01cadf5d1b044e663dfa862bdc14e4a80fe76e6ccd62d81a66a920bf1e5072ccfe49d9c10a55ce112e512fe2ce48b5", 1),
        ("16ac28fbeb8e9fa3e408aad1f0687c8b204fa8459265817b9fe05f29c8f4cb3b4a81128e02e4a961a8931a18f742cbf0b7345845609b5cfaa0e6445935d01717d34", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", 1),
        ("261e7d6d20ba6b4abc39fae0771a060d37d37bf48482e639ba2f391d006f242a62d0e05c1d28dad24638be1e2e15b3568199b8b07941e901df6ab150bddda43756c60dc88317d03e3969cae4bc14097f0398aa537cf5e70419cbfb295b2616f82c2aa2ee780db6b02dd4f652971a3ef1bad6e2ff7814240c2523f2ae415df4f5bdbeb", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", -1),
        ("197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984874", "197f69d016a44ac2c2b3f33a9cbcc59f0fc27e0aa59f72ff1c1e4201467eeb251976abe8c60445644eb272ea1d423526a672a28ceb092ee24f08b4f39e712984875", 1),
        ("53201e0f5c468d338127426ab27d05defa5f02e918be973a97d6313dd0d25db829e7f5790b9bbf4457b83b1909bb1be0a87a6d9b37cd20f80d57d90e9f84eb90c50c71e1237ad6643da295612f7566c740f4af6b6081692001dc635d1c6f8a96d7b49acc59e8bd64bf9ed5893607f5e5ba0111b427513417ee2b21598ee9fdfb", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", 1),
        ("4d38bb7b9bb8b762f8ed62417a9685531dba5434d7f4b6c738d8e99d0e8c6496c16b182b5df84ca47248428d37c42b2cf01e8d4002de7b0c75ba59c9014246bae8f9752bf48332f38fd982630cd1e1c37b5cb3ad845d4640d131044066f672ce1bb90527aaa19549240b6779fb6c223bbacbbf381f17cd9260bc795f77cb82c7132fd14bec2f2c4528bfaaa393a178a67004d43c54f97831c98a60e456ebdefaf392dd19112e47c19f78926504f54693c14deb5bb7670df590f271439188deb3ca0d6615a299edae130de3893255c242e5b3faa6d7991dc7ad222ecda559705b3beaf902c6fafc4dee7c1faffe684a474eb955bcb8fd3083bc89740d93c537f2", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", 1),
        ("9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11e", "9b7b9b81fc01fec3271906924bca2b208706aa7d20d1a7170ecfe68258e6980a4bd9f5a5c7be65fad55b8b9d197bdcc8d3b68c0f4a2f2c68873bf95d21ad185f73163bf57394a9b237be271865c2af8afe72caa22329037de24513510492c063fad5cb2e20cb51d035db09801a74886ccd897ba62139283595ff2255b73cc11f", -1),
        ("98776f863c74a863b859270a498cd465f35fbd549c1c1cadf1f2f01919db2287f94a9b7b2d8f078f03ed80246f7f1cd1218384a8df73432530f139d9e3f141b8cf98a159109f035a83e5384ad37de0e792f669b89a189a43ae150b21a007786067b78e84bb04c689035e1fac0ae7739fa24666a31c9f721c8b89e8bf17ecea09", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", 0),
        ("aaba65380cda8fbff9d381278536de363e4b7513409e9443ddd86a1dd7facc7cfa429d5cd59a7a7b5cb6ad2396bed3ebb7d6c27cf7986c4846086d8b9d3743285f19b66c71d12149ada06504a8a148c633c65191662942b0624cf777fb76517720b096eac25cadc9a745a0949a4b8d7d421b7b4e4c07bdd6e32a056ade9660181e36b661168b1d83021f181b856d576ad64aa068d56e22d85af15be0611814039ee29f4c4df81eafbc82100628549716e839d053684fed1f32f1059b5b78287db19ceb1600073d1cb0b65bc4d77bcef5b02562f5a305a4aae00e676394fe413c9c8e9330f62c9ce6193cbc3fe572cb6f12175d04aafb4637875fc60ca83e2231", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", 1),
        ("cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88a", "cffa14009aafb8653889ea2540d706b76270e9d117f2bf135581ec1c98a2f2515a09c7bdce43beab643ef15d5134bd950f97f26f8d7115e3326ec7489abe780d64c66b847eb3ddad05d76d1f4d9a79740b2ce06847ae562a6ce41913aa5c94eae15b18e35152e269f0bf47c30d8c14d71208af84403946e4d2bb96b317f7c88b", -1),
        ("9bc69b5c956a1324ddddd1d93a3e2b7134b3809ec28c2ebb94b935243a3afa9704a6d1ebab260dbcc7cfcbda9f58cc097c9d97aa5e719cd55551e47083d6993a0fb9549aac43a4bfcf7956b4932c83adbb3f5b7229025a345279dd42e184032ec167811d19f5f5c3db16ff063789c586c3e9580d371bd16fdb91967d43559420", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", -1),
        ("8ef67abc4b95368f5788bcc0ce23acb9121d307ce500562b8e48df3f8fc1ccbf87d3113f58ef99a61a2c1d9235193c26dd6a1713ea4d30af6e760a034f4221a5f10ad5faaa55fcd63b7e3d16359075fffc7bd96e327263eda214dd456862292893615a0797db168a7e93ddef75bb0bfea31047ca93e62f08dc3ca5681bf5d242018491b16a1507bd9beb3542dc1fa08130969a2be83c72ce67b3fa37d1a43e53c670a88f0eb7e331cd393c4de0caa897b6bc519c02ed51e57d60666dbc305dac7870588fec7d15a4c0cfbfb48685006a27cefeb06d825a7253a812027e7f5a94b6e81f50784a9a9159406727ac4de65b42d280b0a1e07351d093c894bf0f8d3a", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", 0),
        ("dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55e", "dfd434e12b6c201280b5a2c4a84aa2586e87172873a3d6e0f3590c1ce25b4b5ffd44c3356a69c2f8388851d0db87e7548b16c064cb5a64235d604a9e87f01c93bded1b75891a885723f00e66ab1825f7dbf58c04d7da131f26c9b563d5459ceb0c9b0b110af70c767ab8b47a4029bf138275db973c072e640956aed70624f55f", -1),
        ("6d711a6962b72ca27cd7c741d69d69902c0d2ef908f4911b2f0579b175c39f1d648fb796bf3097c316f5a7f42da363ad8efca78b860f780b97f9a6208e8951d5fcdf1b31cb875fbe8bd58f3e50afc54ffe8d06182d5164db93f58749d03cae674d0a3765c366e66e251d0dba7adb1ee30083779c0725e3af1c8d9f591f59a53d", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", 1),
        ("17963ddf237ce0ecc4da248b44f666dd74bfe36f1f6bc837d75616e9ae0e52eae7e4c051429004ef1c88fea3cf922e0448fde6d48a4c52107035489c8e5cbafbee0cd0873f40aeb51e7586a7e8ca5759e4f61608fb9d456248d6f03414cd42141a02567b0acb6ced83fdd64dd3f06bee6b5ad4a27378ebe69f26b165fa3cac643465ec905145a8612fe10d27763e0bdc7808dccc18b344c90b92e482a8a2ab6912152f45eb34e46e21cfd49438656d68b43ba9a4b9f6d77d6d45a2240ec28dbdacf1421f95f83b7b6fcccd1f52e2049f7a973b64d28fc0d999d2edb572411166e68ba95f55020d7c729e979b745c47b8cfd930afb8fb9fb536f5646213e15c2c", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", 1),
        ("a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139716", "a20295f47f3d794b692ddee446afbc48389118927fc50f2c058d319df1a2b889ec2617bf16934f2df25a6a491000dc629b89d729e1507b4624ea2ea9d77391e17cffb7067f9974f6971ee2f181832ae5eecd18705696ecc75888196b67e76790254b15c22a26d4400e199860bc712c6dafdf0e325747db884e4c958a98139717", -1),
        ("7c453669df9f75cf0c9ead528695ccdcc15e6002f6b32f76e740fec6925db2e1a3a57495a04e67ec52ae2a8216f7655595a60d6ea0b639c1361552f137056a9684e56e6f3469cabc9ca2a6a0e90ddb4bbac8d95c6271191b939e3c4b590e243c8a78cf96b55e754e5ad691181c42bfda894a7dea065253ed1eea627badcd9d80", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", 1),
        ("d37c1c26a89e630919e5ec863222e184849fc23b067dcf7fcaff6564d9dd5001d43f294720289285f0407e3ab4fab556d247cb949118191d92c6361513bedd85de1b7f6bb5f51b79a431cbe777dd3ded2466f2207dead1abc6a7a19ef0635fb43ce03c8a4cf0b4a35425baa714b497d5b3974cefe6fdafd71edbed732318b4fb462046f5ac98acc37c8693ec2987f1b773b5e702536c37abeb07b7dc1bda0131f640f2926ac6d39487d4b6f87a262d1083736bd4af80f445386910c61ed634bb9ca10980e3e6c29e5373cbe942a506a471c5f597938358e0fc62381faaa89ae2d6e17d5c8e2f5ebcb3b14096c091c97147aa1d0abe793ac030c78427073969c2", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", -1),
        ("ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2c", "ba00833faf06a90d15f73f79591f000e8db2ccce78b559c36a203e19dac00ffe5f371987ee4daf0b81affddc679a21d9252b6e7cee92f74e6133a1495536cc1414f72cfc8f72cf6b10a9701dfb82582a6ed824cfdda0e5924af94778f098963836c911b565c17da6333960b1d8b353cea4d6b25e379fb5f0ff4bb00781309f2d", 1),
        ("0", "1", 1),
        ("5", "1", 1),
        ("0", "f", 0),
        ("1", "f", 1),
        ("2", "7", 1),
        ("2", "9", 1),
        ("2", "b", -1),
        ("2", "d", -1),
        ("2", "f", 1),
        ("3", "c9", 0),
        ("ff", "ff", 0),
        ("10001", "fffffffffffffffb", 1),
    ];

    #[test]
    fn jacobi_matches_gmp_vectors() {
        for &(a_hex, n_hex, expected) in GMP_JACOBI_VECTORS {
            let a = biguint_from_hex(a_hex);
            let n = biguint_from_hex(n_hex);
            assert_eq!(
                jacobi(&a, &n),
                Some(expected),
                "jacobi({a_hex}, {n_hex}) != {expected}"
            );
        }
    }

    #[test]
    fn jacobi_matches_euler_criterion_on_primes() {
        // For odd prime p the Jacobi symbol is the Legendre symbol, and
        // Euler's criterion computes it independently: a^((p-1)/2) mod p is
        // 1 for residues, p-1 for non-residues, 0 when p | a. mod_pow rides
        // the Montgomery kernels, a code path disjoint from jacobi's
        // shift-and-reciprocity loop. The large primes are the Mersenne
        // primes M89, M107, M127.
        let mut rng = SplitMix64 { state: 0x3c3c_3c3c };
        let mut primes: Vec<BigUint> = [3u64, 5, 7, 11, 13, 65_537, 2_147_483_647]
            .iter()
            .map(|&p| BigUint::from_u64(p))
            .collect();
        primes.push(mersenne(89));
        primes.push(mersenne(107));
        primes.push(mersenne(127));

        for p in &primes {
            let exponent = p.sub_ref(&BigUint::one());
            let mut half = exponent.clone();
            half.shr_bits(1);
            for _ in 0..12 {
                let a = draw_below(&mut rng, p);
                let euler = mod_pow(&a, &half, p);
                let expected = if euler.is_one() {
                    1
                } else if euler == p.sub_ref(&BigUint::one()) {
                    -1
                } else {
                    assert!(euler.is_zero(), "Euler criterion out of range");
                    0
                };
                assert_eq!(jacobi(&a, p), Some(expected), "disagrees for p={p:?}");
            }
        }
    }

    #[test]
    fn jacobi_is_multiplicative_and_periodic() {
        let mut rng = SplitMix64 { state: 0x5151_5151 };
        let bound = BigUint::from_u128(1 << 80);
        for _ in 0..40 {
            let mut n1 = draw_below(&mut rng, &bound);
            let mut n2 = draw_below(&mut rng, &bound);
            if !n1.is_odd() {
                n1 = n1.add_ref(&BigUint::one());
            }
            if !n2.is_odd() {
                n2 = n2.add_ref(&BigUint::one());
            }
            let a = draw_below(&mut rng, &n1);
            let b = draw_below(&mut rng, &n1);

            // Multiplicative in the top argument: (ab/n) = (a/n)(b/n).
            assert_eq!(
                jacobi(&a.mul_ref(&b), &n1).unwrap(),
                jacobi(&a, &n1).unwrap() * jacobi(&b, &n1).unwrap()
            );
            // Multiplicative in the bottom: (a/n1n2) = (a/n1)(a/n2).
            assert_eq!(
                jacobi(&a, &n1.mul_ref(&n2)).unwrap(),
                jacobi(&a, &n1).unwrap() * jacobi(&a, &n2).unwrap()
            );
            // Periodic in the top argument: (a + n / n) = (a/n).
            assert_eq!(jacobi(&a.add_ref(&n1), &n1), jacobi(&a, &n1));
        }
    }

    #[test]
    fn jacobi_two_supplement_follows_n_mod_8() {
        // (2/n) = +1 for n = 1, 7 (mod 8) and -1 for n = 3, 5 (mod 8).
        let two = BigUint::from_u64(2);
        for n in (3u64..200).step_by(2) {
            let expected = match n % 8 {
                1 | 7 => 1,
                3 | 5 => -1,
                _ => unreachable!("n is odd"),
            };
            assert_eq!(jacobi(&two, &BigUint::from_u64(n)), Some(expected));
        }
    }

    #[test]
    fn jacobi_edge_cases() {
        // Undefined for even or zero moduli.
        assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::zero()), None);
        assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::from_u64(10)), None);
        // Empty-product convention and the zero cases.
        assert_eq!(jacobi(&BigUint::zero(), &BigUint::one()), Some(1));
        assert_eq!(jacobi(&BigUint::from_u64(9), &BigUint::one()), Some(1));
        assert_eq!(jacobi(&BigUint::zero(), &BigUint::from_u64(9)), Some(0));
        assert_eq!(
            jacobi(&BigUint::from_u64(21), &BigUint::from_u64(7)),
            Some(0)
        );
    }

    #[test]
    fn miller_rabin_witness_primitive() {
        // 341 = 11 * 31 is the classic Fermat 2-pseudoprime; base 3 is a
        // Miller-Rabin witness for it, while no witness testifies against a
        // genuine prime.
        let n341 = BigUint::from_u64(341);
        assert!(miller_rabin_witness(&n341, &BigUint::from_u64(3)));
        let p = BigUint::from_u64(1_000_000_007);
        for base in [2u64, 3, 5, 7, 11] {
            assert!(!miller_rabin_witness(&p, &BigUint::from_u64(base)));
        }

        // Trivial witnesses (0, ±1 mod n) can never testify.
        assert!(!miller_rabin_witness(&n341, &BigUint::zero()));
        assert!(!miller_rabin_witness(&n341, &BigUint::one()));
        assert!(!miller_rabin_witness(&n341, &BigUint::from_u64(340)));
        assert!(!miller_rabin_witness(&n341, &n341));

        // Even and degenerate candidates are composite by inspection.
        assert!(miller_rabin_witness(
            &BigUint::from_u64(100),
            &BigUint::from_u64(3)
        ));
        assert!(miller_rabin_witness(&BigUint::one(), &BigUint::from_u64(3)));
    }

    #[test]
    fn gcd_small_values() {
        let lhs = BigUint::from_u64(48);
        let rhs = BigUint::from_u64(18);
        assert_eq!(gcd(&lhs, &rhs), BigUint::from_u64(6));
    }

    #[test]
    fn lcm_small_values() {
        let lhs = BigUint::from_u64(60);
        let rhs = BigUint::from_u64(52);
        assert_eq!(lcm(&lhs, &rhs), BigUint::from_u64(780));
    }

    #[test]
    fn modular_exponentiation_small_values() {
        let base = BigUint::from_u64(7);
        let exponent = BigUint::from_u64(560);
        let modulus = BigUint::from_u64(561);
        assert_eq!(mod_pow(&base, &exponent, &modulus), BigUint::from_u64(1));
    }

    #[test]
    fn miller_rabin_rejects_composites() {
        assert!(!is_probable_prime(&BigUint::from_u64(561)));
        assert!(!is_probable_prime(&BigUint::from_u64(341)));
        assert!(!is_probable_prime(&BigUint::from_u64(221)));
    }

    #[test]
    fn miller_rabin_accepts_primes() {
        assert!(is_probable_prime(&BigUint::from_u64(65_537)));
        assert!(is_probable_prime(&BigUint::from_be_bytes(&[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc5
        ])));
    }

    #[test]
    fn miller_rabin_rejects_empty_witness_sets() {
        assert!(!is_probable_prime_with_bases(
            &BigUint::from_u64(65_537),
            &[]
        ));
    }

    #[test]
    fn modular_inverse_small_values() {
        assert_eq!(
            mod_inverse(&BigUint::from_u64(11), &BigUint::from_u64(16)),
            Some(BigUint::from_u64(3))
        );
        assert_eq!(
            mod_inverse(&BigUint::from_u64(23), &BigUint::from_u64(46)),
            None
        );
    }
}
