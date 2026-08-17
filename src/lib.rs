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
//! [`Gf2m`](crate::finite_field::Gf2m) for binary extension fields GF(2^m), [`PolyZ`](crate::polynomial::PolyZ)/[`PolyMod`](crate::polynomial::PolyMod) for
//! univariate polynomials, and [`lll_reduce`](crate::lattice::lll_reduce) for lattice basis reduction.
//!
//! The arithmetic and number theory are deterministic functions of their
//! inputs; the sampling routines (`random_below` and friends) are driven by a
//! caller-supplied generator and choose no entropy source of their own.
//!
//! Two properties define the intended use:
//!
//! - **Variable-time, for non-secret data.** Operations take data-dependent
//!   paths; do not use them where timing must not leak secrets.
//! - **Not a secret-scrubbing or constant-time type.** Nothing is wiped:
//!   values live in ordinary heap buffers, freed memory keeps its contents,
//!   and `Debug` prints every limb. Cryptographic memory hygiene and
//!   constant-time operation are out of scope, and a consumer that handles
//!   key material adds them at that layer with a purpose-built representation
//!   rather than relying on anything here.
//!
//! Safety policy: `#![forbid(unsafe_code)]` crate-wide, with no exceptions —
//! `forbid` rather than `deny` precisely because an inner `allow` cannot lift
//! it, so the guarantee is enforced by the compiler against the crate's own
//! code rather than being a default it could override. `#![deny(missing_docs)]`
//! holds every public item to a doc comment, and `MANUAL.md` carries a worked,
//! test-pinned example for each.
//!
//! Targets: portable. Every place that turns a limb count into a bit index
//! goes through one checked multiplication, so an operand too wide to index
//! on the target refuses rather than wrapping — on a 32-bit `usize` that
//! boundary is reachable, at operands of roughly 537 MB for `len · 64` and
//! 268 MB for `len · 128`, and on a 64-bit one it is not. There is no
//! target-width rejection: 32-bit builds are supported and gated in CI.
//!
//! Minimum supported Rust version: 1.87, the release that stabilized
//! `u64::is_multiple_of` and `usize::is_multiple_of`, which the kernels use
//! throughout. The MSRV is recorded as `rust-version` in `Cargo.toml`.
//!
//! ```
//! use rump::{BigUint};
//! use rump::modular::{mod_pow};
//! use rump::number_theory::{is_probable_prime, jacobi};
//!
//! let p = BigUint::from_u64(1_000_000_007);
//! assert!(is_probable_prime(&p));
//!
//! // Fermat: a^(p-1) ≡ 1 (mod p) for prime p.
//! let a = BigUint::from_u64(31_337);
//! let e = p.sub(&BigUint::one());
//! assert!(mod_pow(&a, &e, &p).is_one());
//!
//! // Euler's criterion, read off the Jacobi symbol.
//! assert_eq!(jacobi(&a, &p), Some(1));
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

// Implementation modules are private; every public path below is a facade, so
// each exported item has exactly one public path, as NAMES.md requires.
mod bigint;
mod gf2m;
#[path = "lattice.rs"]
mod lattice_impl;
#[path = "number_theory.rs"]
mod number_theory_impl;
mod poly;
#[path = "random.rs"]
mod random_impl;

pub use bigint::{BigInt, BigUint, Sign};

/// Machine-word helpers and the parse error for the integer types.
pub mod integer {
    pub use crate::bigint::{ParseBigIntError, WordReciprocal};
}

/// Residue-ring arithmetic: the fixed-modulus contexts and the modular
/// operations that are free functions.
pub mod modular {
    pub use crate::bigint::{BarrettContext, ModulusError, MontgomeryContext};
    pub use crate::number_theory_impl::{
        mod_inverse, mod_inverse_batch, mod_inverse_u64, mod_pow, mod_sqrt, mod_sqrt_prime_power,
    };
}

/// Divisibility, symbols, primality, reconstruction, and batching.
pub mod number_theory {
    pub use crate::number_theory_impl::{
        crt_combine, gcd, gcd_extended, gcd_u64, is_probable_prime, is_probable_prime_bpsw,
        is_strong_lucas_probable_prime, jacobi, jacobi_u64, kronecker, lcm, legendre,
        miller_rabin_with_bases, miller_rabin_witness, primes_below, product_tree,
        rational_reconstruct, rational_reconstruct_bounded, remainder_tree, remove_factor,
        smooth_parts, valuation, ProductTree, SmoothnessBase, SmoothnessBaseError,
    };
}

/// Univariate polynomials over ℤ and over a residue ring.
pub mod polynomial {
    pub use crate::poly::{PolyMod, PolyZ, MAX_ENUMERATED_ROOTS};
}

/// Binary extension fields GF(2^m).
pub mod finite_field {
    pub use crate::gf2m::Gf2m;
}

/// Lattice basis reduction.
pub mod lattice {
    pub use crate::lattice_impl::{
        gauss_reduce_weighted, lll_reduce, lll_reduce_delta, ReductionError,
    };
}

/// Sampling, driven entirely by a caller-supplied byte source.
pub mod random {
    pub use crate::random_impl::{
        random_below, random_coprime_below, random_nonzero_below, random_probable_prime,
        RandomSource,
    };
}
