//! Multiprecision integers, modular arithmetic, number theory, polynomials,
//! finite fields, lattices, and sparse linear algebra. Implementations cite
//! their sources at the relevant definitions; `CITATIONS.md` collects them.
//!
//! Operations are variable-time and intended for non-secret data. Sampling
//! uses a caller-supplied generator; the crate chooses no entropy source.
//!
//! The default build forbids unsafe code and does not wipe buffers. The
//! `wipe` feature permits volatile scrubbing of live integer limbs, abandoned
//! limbs on in-place shrink, arithmetic workspaces, and sampler buffers.
//! Spare capacity and buffers freed by reallocation are not covered. `Debug`
//! exposes values, and wiping does not make arithmetic constant-time.
//!
//! The crate supports 32- and 64-bit targets and requires Rust 1.87 or later.
//! Index-size calculations are checked for overflow.
//!
//! ```
//! use rump::{BigUint};
//! use rump::modular::{mod_pow};
//! use rump::number_theory::{is_probable_prime, jacobi};
//!
//! let p = BigUint::from_u64(1_000_000_007); // prime
//! assert!(is_probable_prime(&p));
//!
//! // Fermat: a^(p-1) ≡ 1 (mod p) for prime p.
//! let a = BigUint::from_u64(31_337); // arbitrary
//! let e = p.sub(&BigUint::one());
//! assert!(mod_pow(&a, &e, &p).is_one());
//!
//! // Euler's criterion, read off the Jacobi symbol.
//! assert_eq!(jacobi(&a, &p), Some(1));
//! ```

#![cfg_attr(not(feature = "wipe"), forbid(unsafe_code))]
#![cfg_attr(feature = "wipe", deny(unsafe_code))]
#![deny(missing_docs)]

// Public facades follow the API ownership recorded in NAMES.md.
mod bigint;
#[path = "gf2.rs"]
mod gf2_impl;
mod gf2m;
#[path = "gfp.rs"]
mod gfp_impl;
#[path = "lattice.rs"]
mod lattice_impl;
mod machine;
mod modular_fixed;
#[path = "number_theory.rs"]
mod number_theory_impl;
mod parallel;
mod poly;
#[path = "random.rs"]
mod random_impl;
mod scrub;

pub use bigint::{BigInt, BigUint, Sign};

/// Machine-word helpers and the parse error for the integer types.
pub mod integer {
    pub use crate::bigint::{ParseBigIntError, WordReciprocal};
}

/// Residue-ring arithmetic: the fixed-modulus contexts and the modular
/// operations that are free functions.
pub mod modular {
    pub use crate::bigint::{
        BarrettContext, ContextMismatch, ModulusError, MontgomeryContext, MontgomeryResidue,
        MontgomeryScratch,
    };
    pub use crate::modular_fixed::{Montgomery128, Montgomery64, Residue128, Residue64};
    pub use crate::number_theory_impl::{
        mod_inverse, mod_inverse_batch, mod_inverse_u128, mod_inverse_u64, mod_pow, mod_sqrt,
        mod_sqrt_prime_power,
    };
}

/// Divisibility, symbols, primality, reconstruction, and batching.
pub mod number_theory {
    pub use crate::modular_fixed::{is_prime_u64, is_strong_probable_prime_u64};
    pub use crate::number_theory_impl::{
        crt_combine, crt_combine_balanced, crt_combine_u64, gcd, gcd_extended, gcd_u128, gcd_u64,
        is_lucas_probable_prime, is_prime_aks, is_probable_prime, is_probable_prime_bpsw,
        is_strong_lucas_probable_prime, jacobi, jacobi_u64, kronecker, lcm, legendre,
        miller_rabin_with_bases, miller_rabin_witness, primes_below, primes_past, product_tree,
        rational_reconstruct, rational_reconstruct_bounded, remainder_tree, remove_factor,
        smooth_parts, valuation, CrtBasis, ProductTree, SmoothnessBase, SmoothnessBaseError,
    };
}

/// Univariate polynomials over ℤ and over a residue ring.
pub mod polynomial {
    pub use crate::poly::{
        ApproximateRoot, FactorRealError, HenselSquareRoot, PolyMod, PolyZ, RealFactorization,
        RealRootError, MAX_ENUMERATED_ROOTS,
    };
}

/// Linear algebra over GF(2): null space, pruning, structured elimination, Block Lanczos.
pub mod gf2 {
    pub use crate::gf2_impl::{
        block_lanczos_dependencies, block_lanczos_dependencies_sparse, dense_null_space,
        filter_merge, prune_singletons, FilteredMatrix, MatrixBytesError, PrunedMatrix,
        SparseMatrix,
    };
}

/// Linear algebra over a large prime field GF(l): sparse matrices with small
/// integer entries, and Wiedemann's algorithm.
pub mod gfp {
    pub use crate::gfp_impl::{kernel_vector, minimal_polynomial, Field, Kernel, SparseMatrix};
}

/// Binary extension fields GF(2^m).
pub mod finite_field {
    pub use crate::gf2m::Gf2m;
}

/// Lattice basis reduction.
pub mod lattice {
    pub use crate::lattice_impl::{
        closest_vectors_form, gauss_reduce_weighted, lll_reduce, lll_reduce_delta, lll_reduce_form,
        short_vectors_form, Enumeration, EnumerationOutcome, ReductionError,
    };
    pub use crate::poly::bareiss_determinant;
}

/// How many threads the work on a thread may use, and what the machine's
/// threads can do together.
pub mod parallelism {
    pub use crate::machine::{gather_costs, GatherCosts};
    pub use crate::parallel::{budget, with_budget};
}

/// Sampling, driven entirely by a caller-supplied byte source.
pub mod random {
    pub use crate::random_impl::{
        random_below, random_coprime_below, random_nonzero_below, random_probable_prime,
        RandomSource,
    };
}
