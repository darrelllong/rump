# MANUAL

This manual documents every public API in the crate, organized by task. Every
code block below is replicated verbatim in `tests/manual_examples.rs`, so the
examples are compiled and asserted on every `cargo test` — the manual cannot
drift from the code.

## Imports and Naming

The crates.io package is `rust-mp`; the library target is `rump`, so code
always says `use rump::...`:

```toml
[dependencies]
rust-mp = "0.1"
```

```rust
use rump::{
    crt_combine, gcd, gcd_extended, is_probable_prime, is_probable_prime_with_bases, jacobi,
    kronecker, lcm, legendre, miller_rabin_witness, mod_inverse, mod_pow, random_below,
    random_coprime_below, random_nonzero_below, random_probable_prime, sqrt_mod, BigInt, BigUint,
    MontgomeryCtx, Rng, Sign,
};
```

Two properties hold everywhere: operations are **variable-time** (do not use
them where timing must not leak secrets), and every `BigUint` **wipes its
limbs on drop**, with exponentiation workspaces wiped on exit.

## BigUint

Unsigned multiprecision integers on little-endian `u64` limbs.

### Construction and bytes

`zero`, `one`, `from_u64`, and `from_u128` build small values;
`from_be_bytes` / `to_be_bytes` are the one external format — big-endian
bytes, as DER, PEM, and the RFC wire formats use. `to_be_bytes` strips
leading zero bytes (zero encodes as a single `0x00`).

```rust
assert!(BigUint::zero().is_zero());
assert!(BigUint::one().is_one());

let small = BigUint::from_u64(0xFFFF_FFFF_FFFF_FFFF);
let wide = BigUint::from_u128(1u128 << 96);
assert!(small < wide);

let value = BigUint::from_be_bytes(&[0x01, 0x00]);
assert_eq!(value, BigUint::from_u64(256));
assert_eq!(value.to_be_bytes(), vec![0x01, 0x00]);
```

### Predicates and comparison

`is_zero`, `is_odd`, `is_one`, and `bits` (the number of significant bits);
`BigUint` orders with the standard comparison operators.

```rust
let n = BigUint::from_u64(256);
assert_eq!(n.bits(), 9);
assert!(!n.is_odd());
assert!(!n.is_zero());
assert!(BigUint::from_u64(255) < n);
assert_eq!(BigUint::zero().bits(), 0);
```

### Arithmetic

`add_ref` / `sub_ref` / `mul_ref` return new values; `add_assign_ref` /
`sub_assign_ref` work in place. `square_ref` squares, and `sqrt_floor` is
the integer square root (largest `r` with `r² ≤ self`). Subtraction panics
on underflow — the type is unsigned; use [`BigInt`](#signed-integers-bigint-and-sign)
when signs can go negative.

```rust
let a = BigUint::from_u64(1_000);
let b = BigUint::from_u64(37);

assert_eq!(a.add_ref(&b), BigUint::from_u64(1_037));
assert_eq!(a.sub_ref(&b), BigUint::from_u64(963));
assert_eq!(a.mul_ref(&b), BigUint::from_u64(37_000));
assert_eq!(b.square_ref(), BigUint::from_u64(1_369));
assert_eq!(BigUint::from_u64(17).sqrt_floor(), BigUint::from_u64(4));

let mut acc = BigUint::from_u64(1_000);
acc.add_assign_ref(&b);
acc.sub_assign_ref(&b);
assert_eq!(acc, a);
```

### Shifts and bit access

`shl_bits` / `shr_bits` shift by any count; `shl1` / `shr1` are the
single-bit forms the division kernel uses in its hot loop. `bit` tests a
bit, `set_bit` sets one, and `bitxor_assign` is GF(2ᵐ) field addition.

```rust
let mut n = BigUint::from_u64(5);
n.shl_bits(64);
assert_eq!(n, BigUint::from_u128(5u128 << 64));
n.shr_bits(64);
assert_eq!(n, BigUint::from_u64(5));

n.shl1();
assert_eq!(n, BigUint::from_u64(10));
n.shr1();
assert_eq!(n, BigUint::from_u64(5));

assert!(n.bit(0) && n.bit(2) && !n.bit(1));
n.set_bit(4);
assert_eq!(n, BigUint::from_u64(21));

let mut x = BigUint::from_u64(0b1100);
x.bitxor_assign(&BigUint::from_u64(0b1010));
assert_eq!(x, BigUint::from_u64(0b0110));
```

### Division and reduction

`div_rem` returns quotient and remainder (Knuth Algorithm D under the hood);
`modulo` keeps only the remainder; `rem_u64` reduces by a machine word;
`mod_mul` is one-shot modular multiplication. All panic on a zero divisor
or modulus.

```rust
let n = BigUint::from_u64(1_000);
let d = BigUint::from_u64(7);

let (q, r) = n.div_rem(&d);
assert_eq!(q, BigUint::from_u64(142));
assert_eq!(r, BigUint::from_u64(6));
assert_eq!(n.modulo(&d), r);
assert_eq!(n.rem_u64(7), 6);

let product = BigUint::mod_mul(
    &BigUint::from_u64(123),
    &BigUint::from_u64(456),
    &BigUint::from_u64(97),
);
assert_eq!(product, BigUint::from_u64(22)); // 123 · 456 = 56 088 ≡ 22 (mod 97)
```

## Signed integers: BigInt and Sign

A `BigInt` is a `Sign` joined to a `BigUint` magnitude. Construct with
`from_biguint` (non-negative) or `from_parts`; read back with `sign()` and
`magnitude()`; `negated` flips the sign. `add_ref` / `sub_ref` are signed;
`mul_biguint_ref` scales by an unsigned factor. `modulo_positive` maps into
the canonical range `[0, n)` — the piece extended Euclid needs.

```rust
let ten = BigInt::from_biguint(BigUint::from_u64(10));
let minus_three = BigInt::from_parts(Sign::Negative, BigUint::from_u64(3));

assert_eq!(minus_three.sign(), Sign::Negative);
assert_eq!(*minus_three.magnitude(), BigUint::from_u64(3));
assert_eq!(minus_three.negated().sign(), Sign::Positive);

assert_eq!(ten.add_ref(&minus_three), BigInt::from_biguint(BigUint::from_u64(7)));
assert_eq!(
    minus_three.sub_ref(&ten),
    BigInt::from_parts(Sign::Negative, BigUint::from_u64(13))
);
assert_eq!(
    minus_three.mul_biguint_ref(&BigUint::from_u64(4)),
    BigInt::from_parts(Sign::Negative, BigUint::from_u64(12))
);

// −3 ≡ 8 (mod 11), in canonical range.
assert_eq!(
    minus_three.modulo_positive(&BigUint::from_u64(11)),
    BigUint::from_u64(8)
);

// Zero is its own sign; from_parts normalizes it.
assert_eq!(BigInt::zero().sign(), Sign::Zero);
assert_eq!(
    BigInt::from_parts(Sign::Negative, BigUint::zero()).sign(),
    Sign::Zero
);
```

## The Montgomery domain

`MontgomeryCtx::new` precomputes the Montgomery constants for one odd
modulus (`None` for even or zero); `modulus()` returns it. Long computations
encode once, stay in the domain with `mul_mont` / `square_mont`, and decode
at the boundary; `one_mont()` is the encoding of one. `mul`, `square`, and
`pow` are the one-shot forms that convert internally. `pow_encoded` reuses
an already encoded base across exponents.

```rust
let p = BigUint::from_u64(97);
let ctx = MontgomeryCtx::new(&p).expect("97 is odd");
assert_eq!(*ctx.modulus(), p);
assert!(MontgomeryCtx::new(&BigUint::from_u64(100)).is_none()); // even

let a = BigUint::from_u64(5);
let b = BigUint::from_u64(6);

// One-shot operations convert in and out internally.
assert_eq!(ctx.mul(&a, &b), BigUint::from_u64(30));
assert_eq!(ctx.square(&BigUint::from_u64(9)), BigUint::from_u64(81));
assert_eq!(ctx.pow(&a, &BigUint::from_u64(3)), BigUint::from_u64(28)); // 125 mod 97

// Staying in the domain: encode once, multiply cheaply, decode once.
let a_mont = ctx.encode(&a);
let b_mont = ctx.encode(&b);
let product_mont = ctx.mul_mont(&a_mont, &b_mont);
assert_eq!(ctx.decode(&product_mont), BigUint::from_u64(30));
assert_eq!(
    ctx.decode(&ctx.square_mont(&a_mont)),
    BigUint::from_u64(25)
);
assert_eq!(ctx.decode(ctx.one_mont()), BigUint::one());

// Reuse an encoded base across exponents.
assert_eq!(
    ctx.pow_encoded(&a_mont, &BigUint::from_u64(3)),
    BigUint::from_u64(28)
);
```

## Number theory

### Divisibility

`gcd` and `lcm` by Euclid; `gcd_extended` returns the Bézout triple
`(g, s, t)` with `g = a·s + b·t`.

```rust
let a = BigUint::from_u64(240);
let b = BigUint::from_u64(46);

assert_eq!(gcd(&a, &b), BigUint::from_u64(2));
assert_eq!(
    lcm(&BigUint::from_u64(4), &BigUint::from_u64(6)),
    BigUint::from_u64(12)
);

let (g, s, t) = gcd_extended(&a, &b);
let bezout = s.mul_biguint_ref(&a).add_ref(&t.mul_biguint_ref(&b));
assert_eq!(bezout, BigInt::from_biguint(g));
```

### Quadratic-residue symbols

`jacobi(a, n)` for odd `n` (`None` otherwise); `legendre` is the same value
under its prime-modulus name; `kronecker` extends to every modulus,
including even and zero.

```rust
// (2/9) = 1 because 9 ≡ 1 (mod 8); (3/9) = 0 by the shared factor.
assert_eq!(jacobi(&BigUint::from_u64(2), &BigUint::from_u64(9)), Some(1));
assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::from_u64(9)), Some(0));
assert_eq!(jacobi(&BigUint::one(), &BigUint::from_u64(4)), None); // even n

// 2 is a residue mod 7 (3² = 9 ≡ 2).
assert_eq!(legendre(&BigUint::from_u64(2), &BigUint::from_u64(7)), Some(1));

// Kronecker handles even moduli: (5/8) = (5/2)³ = −1.
assert_eq!(kronecker(&BigUint::from_u64(5), &BigUint::from_u64(8)), -1);
assert_eq!(kronecker(&BigUint::one(), &BigUint::zero()), 1); // (1/0) = 1
```

### Modular arithmetic

`mod_pow` for any non-zero modulus (Montgomery when odd), `mod_inverse`
(`None` when the gcd exceeds one), `sqrt_mod` by Tonelli–Shanks (`None` for
non-residues; the result is verified by squaring, so a composite modulus
also yields `None`), and `crt_combine` for Chinese remaindering (`None` when
the moduli are not pairwise coprime).

```rust
let p = BigUint::from_u64(41);

assert_eq!(
    mod_pow(&BigUint::from_u64(3), &BigUint::from_u64(4), &BigUint::from_u64(5)),
    BigUint::one() // 81 ≡ 1 (mod 5)
);

assert_eq!(
    mod_inverse(&BigUint::from_u64(3), &BigUint::from_u64(7)),
    Some(BigUint::from_u64(5)) // 3 · 5 = 15 ≡ 1 (mod 7)
);
assert_eq!(mod_inverse(&BigUint::from_u64(2), &BigUint::from_u64(4)), None);

let root = sqrt_mod(&BigUint::from_u64(2), &p).expect("2 is a residue mod 41");
assert_eq!(BigUint::mod_mul(&root, &root, &p), BigUint::from_u64(2));
assert_eq!(sqrt_mod(&BigUint::from_u64(3), &p), None); // non-residue

// Sunzi's classic: 2 mod 3, 3 mod 5, 2 mod 7.
let x = crt_combine(&[
    (BigUint::from_u64(2), BigUint::from_u64(3)),
    (BigUint::from_u64(3), BigUint::from_u64(5)),
    (BigUint::from_u64(2), BigUint::from_u64(7)),
])
.expect("moduli are pairwise coprime");
assert_eq!(x, BigUint::from_u64(23));
```

### Primality

`is_probable_prime` runs trial division plus Miller-Rabin over the twelve
fixed small-prime bases; `is_probable_prime_with_bases` takes an explicit
base set; `miller_rabin_witness` is the single-round primitive for callers
with their own witness schedule. The fixed bases are for candidates you
generated yourself — an adversary can construct pseudoprimes against any
fixed base set, so untrusted input needs extra candidate-derived witnesses
(as the parent cryptography crate's `is_probable_prime_untrusted` adds).

```rust
assert!(is_probable_prime(&BigUint::from_u64(65_537)));
assert!(!is_probable_prime(&BigUint::from_u64(561))); // Carmichael

// 2047 = 23 · 89 is the smallest strong pseudoprime to base 2 alone:
// the single witness is fooled, the fixed base set is not.
let n = BigUint::from_u64(2_047);
assert!(!miller_rabin_witness(&n, &BigUint::from_u64(2)));
assert!(!is_probable_prime_with_bases(&n, &[2]));
assert!(!is_probable_prime(&n));
assert!(miller_rabin_witness(&n, &BigUint::from_u64(3)));
```

## Random sampling

Implement `Rng` — one method, `fill_bytes` — and every sampler is driven by
it. rump chooses no entropy source: output quality is exactly source
quality, so cryptographic callers must supply a CSPRNG. `random_below`
draws uniformly in `[0, upper)` by rejection, `random_nonzero_below` in
`[1, upper)`, `random_coprime_below` additionally requires coprimality, and
`random_probable_prime` searches for a prime of exactly the requested bit
length.

```rust
/// xorshift64: deterministic and compact. Fine for a manual; NOT a CSPRNG.
struct XorShift64(u64);

impl Rng for XorShift64 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let word = self.0.to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

let mut rng = XorShift64(0x1234_5678_9abc_def0);
let bound = BigUint::from_u64(1_000_003);

let draw = random_below(&mut rng, &bound).expect("bound is non-zero");
assert!(draw < bound);

let nonzero = random_nonzero_below(&mut rng, &bound).expect("bound exceeds one");
assert!(!nonzero.is_zero() && nonzero < bound);

let unit = random_coprime_below(&mut rng, &bound, &BigUint::from_u64(30_030))
    .expect("units exist below the bound");
assert_eq!(gcd(&unit, &BigUint::from_u64(30_030)), BigUint::one());

let prime = random_probable_prime(&mut rng, 64).expect("width of at least 2");
assert_eq!(prime.bits(), 64);
assert!(is_probable_prime(&prime));

// Degenerate bounds answer without consuming entropy.
assert_eq!(random_below(&mut rng, &BigUint::zero()), None);
assert_eq!(random_nonzero_below(&mut rng, &BigUint::one()), None);
assert_eq!(random_probable_prime(&mut rng, 1), None);
```

## Panics

The unsigned types treat impossible requests as programming errors, not
recoverable conditions:

| Call | Panics when |
|---|---|
| `sub_ref` / `sub_assign_ref` | the result would be negative |
| `div_rem` / `modulo` / `rem_u64` | the divisor or modulus is zero |
| `BigUint::mod_mul` / `mod_pow` | the modulus is zero |

Fallible mathematics — a missing inverse, a non-residue, an even Montgomery
modulus, non-coprime CRT moduli — returns `Option` instead.
