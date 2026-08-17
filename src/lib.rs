//! Multiprecision integer arithmetic implemented from the literature.
//!
//! Extracted from [darrelllong/cryptography](https://github.com/darrelllong/cryptography)
//! so the arithmetic can serve non-cryptographic consumers and the crate
//! boundary keeps the API free of cryptography-specific coupling. The kernels
//! are auditable against their sources: Knuth's Algorithm D for division
//! (*TAOCP* vol. 2, §4.3.1),
//! Montgomery multiplication with an explicit public Montgomery domain
//! (Montgomery 1985; Koç, Acar & Kaliski 1996), schoolbook, Karatsuba, and
//! Toom–Cook multiplication, Lehmer's gcd (Knuth §4.5.2, Algorithm L) with
//! subquadratic Half-GCD at scale (Möller, Math. Comp. 77 (2008)), and the
//! Jacobi symbol by quadratic reciprocity (*Handbook of Applied Cryptography*,
//! Algorithm 2.149). Every algorithm carries its citation at its definition,
//! and `CITATIONS.md` collects them.
//!
//! Around that integer core sit three layers with the same discipline:
//! [`Gf2m`] for binary extension fields GF(2^m), [`PolyZ`]/[`PolyModP`] for
//! univariate polynomials, and [`lll_reduce`] for lattice basis reduction.
//!
//! The arithmetic and number theory are deterministic functions of their
//! inputs; the sampling routines (`random_below` and friends) are driven by a
//! caller-supplied generator and choose no entropy source of their own.
//!
//! Two properties define the intended use:
//!
//! - **Variable-time, for non-secret data.** Operations take data-dependent
//!   paths; do not use them where timing must not leak secrets.
//! - **Not a secret-scrubbing or constant-time type.** As cheap defense in
//!   depth every [`BigUint`] volatile-wipes its live limbs on drop, and the
//!   Montgomery exponentiation ladder ([`MontgomeryCtx::pow`] /
//!   `pow_encoded`) wipes its workspaces on exit. That is the extent of it:
//!   spare capacity and buffers freed on reallocation are not wiped, the
//!   in-domain [`MontgomeryCtx::mul_mont`] / `square_mont` keep their
//!   scratch, and `Debug` prints every limb. Cryptographic memory hygiene and
//!   constant-time operation are out of scope; a consumer that handles key
//!   material adds them at that layer.
//!
//! Safety policy: `#![deny(unsafe_code)]` crate-wide; the audited exceptions
//! are the volatile-write scrub helper in `scrub`, which has no safe
//! equivalent — a volatile store requires a raw pointer — and the test probe
//! that verifies the scrub on every buffer-shrinking path by reading the raw
//! tail back. `#![deny(missing_docs)]` holds every public item to a doc
//! comment, and `MANUAL.md` carries a worked, test-pinned example for each.
//!
//! Targets: the limb layout and the index arithmetic that sizes it — `bits`,
//! and the `R²`/Karatsuba bit shifts that scale a limb count by 128 — assume a
//! 64-bit `usize`. rump is developed and tested on 64-bit hosts and is not
//! supported on 32-bit targets, where a multi-hundred-megabyte operand could
//! overflow a `usize` index and land a shift in the wrong place. A
//! `compile_error!` enforces this rather than leaving it to the prose: such a
//! build fails outright instead of misindexing at run time.
//!
//! Minimum supported Rust version: 1.87, the release that stabilized
//! `u64::is_multiple_of` and `usize::is_multiple_of`, which the kernels use
//! throughout. The MSRV is recorded as `rust-version` in `Cargo.toml`.
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

// The 64-bit assumption stated in the crate documentation above, enforced.
// `bits()` scales a limb count by 64 and the `R²` and Karatsuba paths scale
// one by 128; on a 32-bit `usize` those products overflow for operands in the
// hundreds of megabytes, which indexes and shifts wrongly rather than failing.
#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "rump requires a 64-bit target: limb indexing and the R²/Karatsuba bit \
     shifts scale a limb count by 64 and 128 and assume those products fit a \
     `usize`. 32-bit and 16-bit targets are not supported."
);

mod bigint;
mod gf2m;
mod lattice;
mod number_theory;
mod poly;
mod random;
mod scrub;

pub use bigint::{BarrettCtx, BigInt, BigUint, MontgomeryCtx, ParseBigIntError, Sign};
pub use gf2m::Gf2m;
pub use lattice::{lll_reduce, lll_reduce_delta};
pub use number_theory::{
    crt_combine, gcd, gcd_extended, gcd_u64, is_probable_prime, is_probable_prime_bpsw,
    is_probable_prime_with_bases, is_strong_lucas_probable_prime, jacobi, jacobi_u64, kronecker,
    lcm, legendre, miller_rabin_witness, mod_inverse, mod_inverse_batch, mod_inverse_u64, mod_pow,
    primes_below, product_tree, rational_reconstruct, rational_reconstruct_bounded, remainder_tree,
    remove_factor, smooth_parts, sqrt_mod, sqrt_mod_prime_power, valuation, ProductTree,
};
pub use poly::{PolyModP, PolyZ, MAX_ROOT_LEVEL};
pub use random::{
    random_below, random_coprime_below, random_nonzero_below, random_probable_prime, Rng,
};
