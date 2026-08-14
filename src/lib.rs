//! Multiprecision integer arithmetic implemented from the literature.
//!
//! Extracted from [darrelllong/cryptography](https://github.com/darrelllong/cryptography)
//! so the arithmetic can serve non-cryptographic consumers and the crate
//! boundary keeps the API honest. The kernels are auditable against their
//! sources: Knuth's Algorithm D for division (*TAOCP* vol. 2, §4.3.1),
//! Montgomery multiplication with an explicit public Montgomery domain
//! (Montgomery 1985; Koç, Acar & Kaliski 1996), schoolbook, Karatsuba, and
//! Toom–Cook multiplication, Lehmer's gcd (Knuth §4.5.2, Algorithm L) with
//! subquadratic Half-GCD at scale (Möller, Math. Comp. 77 (2008)), and the
//! Jacobi symbol by quadratic reciprocity (*Handbook of Applied Cryptography*,
//! Algorithm 2.149). Every algorithm carries its citation at its definition.
//!
//! The arithmetic and number theory are deterministic functions of their
//! inputs; the sampling routines (`random_below` and friends) are driven by a
//! caller-supplied generator and choose no entropy source of their own.
//!
//! Two properties carried over from the parent crate:
//!
//! - **Variable-time.** Operations take data-dependent paths; do not use
//!   them where timing must not leak secrets.
//! - **Scrubbed memory.** Every [`BigUint`] wipes its limbs on drop, and the
//!   Montgomery exponentiation ladder wipes its workspaces on exit, so
//!   values do not linger in freed heap memory.
//!
//! Safety policy: `#![deny(unsafe_code)]` crate-wide; the audited
//! exceptions are the volatile-write scrub helper in `scrub`, which cannot
//! be expressed in safe Rust, and the test probe that verifies the scrub
//! on every buffer-shrinking path by reading the raw tail back. `#![deny(missing_docs)]` holds every public item
//! to a doc comment, and `MANUAL.md` carries a worked, test-pinned example
//! for each.
//!
//! ```
//! use rump::{is_probable_prime, jacobi, mod_pow, BigUint};
//!
//! let p = BigUint::from_u64(1_000_000_007);
//! assert!(is_probable_prime(&p));
//!
//! // Fermat: a^(p-1) ≡ 1 (mod p) for prime p.
//! let a = BigUint::from_u64(31_337);
//! let e = p.sub_ref(&BigUint::one());
//! assert!(mod_pow(&a, &e, &p).is_one());
//!
//! // Euler's criterion, read off the Jacobi symbol.
//! assert_eq!(jacobi(&a, &p), Some(1));
//! ```

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod bigint;
mod gf2m;
mod number_theory;
mod random;
mod scrub;

pub use bigint::{BigInt, BigUint, MontgomeryCtx, Sign};
pub use gf2m::Gf2m;
pub use number_theory::{
    crt_combine, gcd, gcd_extended, is_probable_prime, is_probable_prime_bpsw,
    is_probable_prime_with_bases, is_strong_lucas_probable_prime, jacobi, kronecker, lcm, legendre,
    miller_rabin_witness, mod_inverse, mod_pow, rational_reconstruct, rational_reconstruct_bounded,
    sqrt_mod,
};
pub use random::{
    random_below, random_coprime_below, random_nonzero_below, random_probable_prime, Rng,
};
