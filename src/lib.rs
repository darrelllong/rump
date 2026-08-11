//! Multiprecision integer arithmetic implemented from the literature.
//!
//! Extracted from [darrelllong/cryptography](https://github.com/darrelllong/cryptography)
//! so the arithmetic can serve non-cryptographic consumers and the crate
//! boundary keeps the API honest. The kernels are auditable against their
//! sources: Knuth's Algorithm D for division (*TAOCP* vol. 2, §4.3.1),
//! Montgomery multiplication with an explicit public Montgomery domain
//! (Montgomery 1985; Koç, Acar & Kaliski 1996), schoolbook and Karatsuba
//! multiplication, and the binary Jacobi symbol (*Handbook of Applied
//! Cryptography*, Algorithm 2.149).
//!
//! Two properties carried over from the parent crate:
//!
//! - **Variable-time.** Operations take data-dependent paths; do not use them
//!   where timing must not leak. The parent crate makes this explicit by
//!   namespace (`vt::`).
//! - **Scrubbed memory.** Every [`BigUint`] wipes its limbs on drop, and the
//!   Montgomery exponentiation ladder wipes its workspaces on exit, so
//!   values do not linger in freed heap memory.
//!
//! Safety policy: `#![deny(unsafe_code)]` crate-wide; the sole audited
//! exception is the volatile-write scrub helper in `scrub`, which cannot be
//! expressed in safe Rust.

#![deny(unsafe_code)]

mod bigint;
mod number_theory;
mod scrub;

pub use bigint::{BigInt, BigUint, MontgomeryCtx, Sign};
pub use number_theory::{
    crt_combine, gcd, gcd_extended, is_probable_prime, is_probable_prime_with_bases, jacobi,
    kronecker, lcm, legendre, miller_rabin_witness, mod_inverse, mod_pow, sqrt_mod,
};
