# MANUAL

This manual documents every public API in the crate, organized by task. Every
code block below is replicated verbatim in `tests/manual_examples.rs`, so the
examples are compiled and asserted on every `cargo test` — the manual cannot
drift from the code.

## Imports and Naming

The crates.io package is `rust-mp`; the library target is `rump`, so code
always says `use rump::...`. `BigInt`, `BigUint`, and `Sign` are at the crate
root; every other item has exactly one module path, listed in
[`NAMES.md`](NAMES.md):

```toml
[dependencies]
rust-mp = "0.3"
```

```rust
use core::num::NonZeroU64;
use rump::finite_field::Gf2m;
use rump::integer::WordReciprocal;
use rump::lattice::{gauss_reduce_weighted, lll_reduce, ReductionError};
use rump::modular::{
    mod_inverse, mod_inverse_batch, mod_inverse_u64, mod_pow, mod_sqrt, mod_sqrt_prime_power,
    BarrettContext, ModulusError, MontgomeryContext, MontgomeryScratch,
};
use rump::number_theory::{
    crt_combine, crt_combine_balanced, gcd, gcd_extended, gcd_u64, is_prime_aks, is_probable_prime,
    is_probable_prime_bpsw, is_strong_lucas_probable_prime, jacobi, kronecker, lcm, legendre,
    miller_rabin_with_bases, miller_rabin_witness, primes_below, product_tree,
    rational_reconstruct, rational_reconstruct_bounded, remainder_tree, remove_factor,
    smooth_parts, valuation, SmoothnessBase,
};
use rump::polynomial::{PolyMod, PolyZ, RealRootError};
use rump::random::{
    random_below, random_coprime_below, random_nonzero_below, random_probable_prime, RandomSource,
};
use rump::{BigInt, BigUint, Sign};
```

Two properties define the intended use. Operations are **variable-time** (do
not use them where timing must not leak secrets), and rump is **for non-secret
data — not a secret-scrubbing or constant-time type by default**. In the
default build nothing is wiped: values live in ordinary heap buffers, freed
memory keeps its contents, and `Debug` prints every limb. The opt-in `wipe`
cargo feature restores the drop-time scrub as cheap defense in depth — every
`BigUint` volatile-wipes its live limbs on drop, the in-place shrink paths
wipe the limbs they abandon, the exponentiation ladder and Montgomery
workspaces wipe on exit, and the samplers wipe drawn bytes — with the old
caveats unchanged: spare capacity and buffers freed on reallocation are not
wiped, and nothing becomes constant-time. Constant-time operation is out of
scope either way, left to a consumer that handles key material.

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

// Fixed-width output pads on the left — the shape share and wire
// serializations want; a value that does not fit panics.
assert_eq!(value.to_be_bytes_padded(4), vec![0x00, 0x00, 0x01, 0x00]);

// Range-pinned callers can read the low bits directly.
let wide = BigUint::from_u128((7u128 << 64) | 9);
assert_eq!(wide.low_u128(), (7u128 << 64) | 9);
assert_eq!(wide.low_bits(64), BigUint::from_u64(9)); // wide mod 2^64
```

### Roots, powers, and bit counts

`sqrt_rem` returns the integer square root with its remainder — Newton's
iteration, certified by the first non-decrease — and `sqrt_floor` is its
root half. `nth_root_floor(k)` generalizes to any k ≥ 1 (panics at k = 0).
`is_square` answers by residue filters and one certified root;
`is_perfect_power` checks one certified root per prime exponent up to the
bit length (on odd operands every prime exponent pays a full root — about
22 ms at 4096 bits). `pow_u64` raises to a machine-word exponent by binary
exponentiation — the helper the root routines certify against. `popcount`
counts set bits; `trailing_zeros` is the 2-adic valuation, `None` for
zero.

```rust
let n = BigUint::from_u64(1_000_000);
let (root, rem) = n.sqrt_rem();
assert_eq!(root, BigUint::from_u64(1_000));
assert!(rem.is_zero());
assert!(n.is_square());
assert!(!BigUint::from_u64(1_000_001).is_square());

assert_eq!(BigUint::from_u64(1_000_000).nth_root_floor(3), BigUint::from_u64(100));
assert!(BigUint::from_u64(729).is_perfect_power()); // 3^6
assert!(!BigUint::from_u64(730).is_perfect_power());

assert_eq!(BigUint::from_u64(3).pow_u64(6), BigUint::from_u64(729));
assert_eq!(BigUint::from_u64(0b1011_0000).popcount(), 3);
assert_eq!(BigUint::from_u64(0b1011_0000).trailing_zeros(), Some(4));
assert_eq!(BigUint::zero().trailing_zeros(), None);
```

### Radix strings

`from_str_radix` / `to_str_radix` convert against any radix from 2 to 36
(digits `0-9a-z`, upper case accepted on input, lower case produced, no
prefixes or whitespace); both panic outside that range, matching the
standard library's contract. `BigInt`'s forms carry an optional leading
`-`. `Display` and `FromStr` are the base-10 special case, so `format!`
with `{}` and `.parse()` work as for machine integers, with two named
divergences: a leading `+` is rejected (the parser reads values, not
literals — the `FromStr` error type is the exported `ParseBigIntError`),
and the `{:x}`/`{:o}`/`{:b}` format traits are not implemented
— hexadecimal and binary come from `to_str_radix`. Power-of-two radices
convert by direct bit packing; the rest run classical word-sized
conversion at small sizes and divide-and-conquer against a ladder of
squared radix powers above the measured crossovers.

```rust
let n = BigUint::from_str_radix("deadbeef", 16).expect("valid hex");
assert_eq!(n, BigUint::from_u64(0xdead_beef));
assert_eq!(n.to_str_radix(16), "deadbeef");
assert_eq!(n.to_string(), "3735928559");
assert_eq!("3735928559".parse::<BigUint>(), Ok(n));

assert_eq!(BigUint::from_str_radix("rump", 36), Some(BigUint::from_u64(1_299_409)));
assert_eq!(BigUint::from_str_radix("12a", 10), None); // invalid digit

let debt = BigInt::from_str_radix("-7", 10).expect("signed parse");
assert_eq!(debt.to_string(), "-7");
assert_eq!("-0".parse::<BigInt>(), Ok(BigInt::zero()));
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

`add` / `sub` / `mul` return new values; `+=` and `-=` on a borrowed
right-hand side work in place. `add_into` / `sub_into` are the
three-operand forms — the result written into `self`, whose buffer is
reused: a long-lived output allocates only until its capacity covers the
result, then never again (the shape of GMP's `mpz_add`).
`Clone::clone_from` likewise copies a value into existing storage.
`square` squares, retaining specialized schoolbook/Karatsuba kernels and a
one-buffer exact NTT square at very large sizes; NTT execution never uses more
contexts than the machine reports. `sqrt_floor` is the integer square root
(largest `r` with `r² ≤ self`). Subtraction panics on underflow — the type is
unsigned; use
[`BigInt`](#signed-integers-bigint-and-sign) when signs can go negative.

```rust
let a = BigUint::from_u64(1_000);
let b = BigUint::from_u64(37);

assert_eq!(a.add(&b), BigUint::from_u64(1_037));
assert_eq!(a.sub(&b), BigUint::from_u64(963));
assert_eq!(a.mul(&b), BigUint::from_u64(37_000));
assert_eq!(b.square(), BigUint::from_u64(1_369));
assert_eq!(BigUint::from_u64(17).sqrt_floor(), BigUint::from_u64(4));

let mut acc = BigUint::from_u64(1_000);
acc += &b;
acc -= &b;
assert_eq!(acc, a);

// Three-operand form: `out`'s storage is reused across calls.
let mut out = BigUint::zero();
out.add_into(&a, &b);
assert_eq!(out, BigUint::from_u64(1_037));
out.sub_into(&a, &b);
assert_eq!(out, BigUint::from_u64(963));
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
`rem` keeps only the remainder; `rem_u64` reduces by a machine word;
`mod_mul` is one-shot modular multiplication. All panic on a zero divisor
or modulus.

```rust
let n = BigUint::from_u64(1_000);
let d = BigUint::from_u64(7);

let (q, r) = n.div_rem(&d);
assert_eq!(q, BigUint::from_u64(142));
assert_eq!(r, BigUint::from_u64(6));
assert_eq!(n.rem(&d), r);
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
`from_biguint` (non-negative), `from_parts`, `from_i64`, or `from_i128`
(total over its range — `i128::MIN`'s magnitude `2^127` is an ordinary
`u128`); read back with `sign()` and
`magnitude()`; `negated` flips the sign. `add` / `sub` are signed,
and `+=` / `-=` on a borrowed right-hand side are their in-place forms, reusing
the magnitude's buffer in every sign combination (nothing panics — the
sign follows the result); `mul` is the full signed product, and
`mul_biguint` scales by an unsigned factor. `div_rem` divides with
remainder, **truncated toward zero** — the C and Rust `/` convention, the
remainder taking the dividend's sign — and panics on a zero divisor;
`abs` returns the magnitude as an owned `BigUint` (prefer `magnitude()`
when a borrow suffices — it lends the same value without the copy).
`rem_euclid` maps into the
canonical range `[0, n)` — the floored remainder, and the piece extended
Euclid needs.

```rust
let ten = BigInt::from_biguint(BigUint::from_u64(10));
let minus_three = BigInt::from_parts(Sign::Negative, BigUint::from_u64(3));

// i128 construction is total: i128::MIN's magnitude is a u128.
assert_eq!(BigInt::from_i128(-3), minus_three);
assert_eq!(
    *BigInt::from_i128(i128::MIN).magnitude(),
    BigUint::from_u128(1u128 << 127)
);

assert_eq!(minus_three.sign(), Sign::Negative);
assert_eq!(*minus_three.magnitude(), BigUint::from_u64(3));
assert_eq!(minus_three.negated().sign(), Sign::Positive);

assert_eq!(ten.add(&minus_three), BigInt::from_biguint(BigUint::from_u64(7)));
assert_eq!(
    minus_three.sub(&ten),
    BigInt::from_parts(Sign::Negative, BigUint::from_u64(13))
);
assert_eq!(
    minus_three.mul_biguint(&BigUint::from_u64(4)),
    BigInt::from_parts(Sign::Negative, BigUint::from_u64(12))
);

// The signed ring: full product, truncated division, absolute value.
assert_eq!(
    minus_three.mul(&ten),
    BigInt::from_parts(Sign::Negative, BigUint::from_u64(30))
);
// div_rem truncates toward zero; the remainder takes the dividend's sign:
// -7 = -3·2 - 1 (floored division would say -7 = -4·2 + 1 instead).
let minus_seven = BigInt::from_parts(Sign::Negative, BigUint::from_u64(7));
let two = BigInt::from_biguint(BigUint::from_u64(2));
let (q, r) = minus_seven.div_rem(&two);
assert_eq!(q, BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)));
assert_eq!(r, BigInt::from_parts(Sign::Negative, BigUint::one()));
assert_eq!(minus_seven.abs(), BigUint::from_u64(7));

// −3 ≡ 8 (mod 11), in canonical range.
assert_eq!(
    minus_three.rem_euclid(&BigUint::from_u64(11)),
    BigUint::from_u64(8)
);

// Zero is its own sign; from_parts normalizes it.
assert_eq!(BigInt::zero().sign(), Sign::Zero);
assert_eq!(
    BigInt::from_parts(Sign::Negative, BigUint::zero()).sign(),
    Sign::Zero
);
```

## Ordinary code: sorting signed integers

The types behave as ordinary Rust values, and three ground rules govern how
they read at a call site. Comparison uses the standard operators — `BigUint`
and `BigInt` implement `Ord` (sign-aware for `BigInt`: negatives below zero,
zero below positives), so `<`, `.max()`, and `slice::sort` all apply.
Arithmetic never does: rump does not overload `+` or `*`, so every
multiprecision operation is an explicit method call (`add`, `mul`,
`negated`, …) and therefore visible in the code that pays for it. Values move
without copying; `Clone` duplicates the limbs;
live limbs on drop as defense in depth (see the scope note above).

A complete example — a bubble sort of signed integers, written exactly as it
would be for any ordered type:

```rust
/// Sort in place by repeated adjacent exchange — pedagogical, not fast.
fn bubble_sort(values: &mut [BigInt]) {
    for pass in 1..values.len() {
        for i in 0..values.len() - pass {
            if values[i] > values[i + 1] {
                values.swap(i, i + 1);
            }
        }
    }
}

// A signed value is built from an unsigned magnitude; negation is a method.
let big = |v: u64| BigInt::from_biguint(BigUint::from_u64(v));
let neg = |v: u64| big(v).negated();

// One operand wider than any machine word: 2^100 + 7.
let mut wide = BigUint::one();
wide.shl_bits(100);
let wide = BigInt::from_biguint(wide.add(&BigUint::from_u64(7)));

let mut values = vec![big(251), neg(40), big(0), wide.clone(), neg(3), big(17)];
bubble_sort(&mut values);
assert_eq!(
    values,
    vec![neg(40), neg(3), big(0), big(17), big(251), wide]
);
```

`Vec::swap` moves the values without touching a limb; each comparison walks
limbs from the top and stops at the first difference; and the sort is
oblivious to whether an element fits in one word or a hundred.

## Barrett contexts

`BarrettContext` is the fixed-modulus reduction context for **either parity**
— the complement to `MontgomeryContext`, which requires an odd modulus. One
division precomputes `μ`; `reduce` then costs two multiplications
(HAC Algorithm 14.42), with `mod_mul`, `mod_square`, and `mod_pow` built
on it. `None` for a modulus below 2.

```rust
let even = BigUint::from_u64(1_000);
let ctx = BarrettContext::new(&even).expect("modulus is at least 2");
assert_eq!(
    ctx.mod_mul(&BigUint::from_u64(123), &BigUint::from_u64(456)),
    BigUint::from_u64(88) // 56 088 mod 1000
);
assert_eq!(
    ctx.mod_pow(&BigUint::from_u64(7), &BigUint::from_u64(13)),
    mod_pow(&BigUint::from_u64(7), &BigUint::from_u64(13), &even)
);
```

## The Montgomery domain

`MontgomeryContext::new` precomputes the Montgomery constants for one odd
modulus, returning `Err(ModulusError::Even)` or `Err(ModulusError::Zero)`
otherwise; `modulus()` returns it. `mul`, `square`, and `pow` are the one-shot
forms that convert in and out internally.

Long computations instead encode once and stay in the domain. `to_residue`
returns an opaque `MontgomeryResidue`, `from_residue` decodes one, `one()` is
the encoding of one, and the domain operations are `mul_residue`,
`square_residue`, `add_residue`, `sub_residue` and `pow_residue` — the
encoding is linear, so domain addition and subtraction are one
compare-and-correct each.

Belonging is by provenance: a context and its clones share an identity, but a
context rebuilt from the same modulus is a different one and refuses residues
it did not make.

The residue is opaque because its invariant cannot be stated in a signature
that takes a bare integer: a domain value is encoded, reduced, and bound to
one context. The type carries all three, so an unencoded, unreduced, or
foreign value cannot reach a kernel. Where the relationship cannot be checked
at compile time — two contexts built at run time — it is checked at run time,
and the operations return `Result<_, ContextMismatch>`.

For loops where the product *is* the loop, every domain operation has a
`_with` form taking a `MontgomeryScratch`, which threads one caller-owned
buffer through a sequence instead of allocating per multiply — measured
per-operation at about 43% for a 64-bit modulus, roughly 25–33% at 256 bits,
and ~20% at 512, falling to 2–3% at 2048 bits and to the edge of measurement
(~1%) at 4096 (the in-tree `mont_workspace_timing` probe reproduces the
numbers with its per-pass spread printed).

```rust
let p = BigUint::from_u64(97);
let ctx = MontgomeryContext::new(&p).expect("97 is odd");
assert_eq!(*ctx.modulus(), p);
assert_eq!(
    MontgomeryContext::new(&BigUint::from_u64(100)),
    Err(ModulusError::Even)
);

let a = BigUint::from_u64(5);
let b = BigUint::from_u64(6);

// One-shot operations convert in and out internally.
assert_eq!(ctx.mul(&a, &b), BigUint::from_u64(30));
assert_eq!(ctx.square(&BigUint::from_u64(9)), BigUint::from_u64(81));
assert_eq!(ctx.pow(&a, &BigUint::from_u64(3)), BigUint::from_u64(28)); // 125 mod 97

// Staying in the domain: encode once, operate cheaply, decode once.
let a_mont = ctx.to_residue(&a);
let b_mont = ctx.to_residue(&b);
let product = ctx.mul_residue(&a_mont, &b_mont).expect("same context");
assert_eq!(ctx.from_residue(&product).expect("same context"), BigUint::from_u64(30));

// Loops thread one scratch buffer through the domain operations: the same
// values, one allocation instead of one per multiply.
let mut scratch = MontgomeryScratch::new();
assert_eq!(
    ctx.mul_residue_with(&a_mont, &b_mont, &mut scratch).expect("same context"),
    product
);

let squared = ctx.square_residue(&a_mont).expect("same context");
assert_eq!(ctx.from_residue(&squared).expect("same context"), BigUint::from_u64(25));

let sum = ctx.add_residue(&a_mont, &b_mont).expect("same context");
assert_eq!(ctx.from_residue(&sum).expect("same context"), BigUint::from_u64(11));

let difference = ctx.sub_residue(&a_mont, &b_mont).expect("same context");
assert_eq!(
    ctx.from_residue(&difference).expect("same context"),
    BigUint::from_u64(96) // 5 − 6 ≡ −1 ≡ 96 (mod 97)
);

assert_eq!(ctx.from_residue(&ctx.one()).expect("same context"), BigUint::one());

// Reuse an encoded base across exponents, staying in the domain.
let cubed = ctx.pow_residue(&a_mont, &BigUint::from_u64(3)).expect("same context");
assert_eq!(ctx.from_residue(&cubed).expect("same context"), BigUint::from_u64(28));

// A residue from another context is refused rather than silently wrong.
let other = MontgomeryContext::new(&BigUint::from_u64(101)).expect("101 is odd");
assert!(other.from_residue(&a_mont).is_err());
```

## Galois fields GF(2^m)

A `Gf2m` is a binary extension field defined by its irreducible polynomial,
encoded as a `BigUint` bit pattern (bit `i` = coefficient of `xⁱ`). The
degree is derived from the polynomial — `new` returns `None` for constants,
and irreducibility is the caller's contract. `add` is an associated function
(XOR needs no modulus); `mul`, `square`, `inverse`, and `half_trace` are
methods, along with `pow`, `div`, `sqrt` (unique — squaring is a bijection),
`trace`, `half_trace` (the odd-degree primitive), and `solve_quadratic`,
which solves `z² + z = c` at every degree and returns `None` exactly when
`Tr(c) = 1`. Multiplication is comb-based over words (*Guide to ECC*,
Algorithm 2.36) with tap-wise reduction, and squaring is linear via a
spread table. `Gf2m::is_irreducible` (Rabin's test) guards the
constructor's contract for untrusted polynomials.

```rust
// GF(2³) with x³ + x + 1.
let field = Gf2m::new(BigUint::from_u64(0b1011)).expect("degree 3");
assert_eq!(field.degree(), 3);
assert_eq!(*field.modulus(), BigUint::from_u64(0b1011));
assert!(Gf2m::new(BigUint::one()).is_none()); // a constant defines no field

// (x + 1)(x² + 1) = x³ + x² + x + 1 ≡ x².
let x_plus_1 = BigUint::from_u64(0b011);
let x2_plus_1 = BigUint::from_u64(0b101);
assert_eq!(field.mul(&x_plus_1, &x2_plus_1), BigUint::from_u64(0b100));
assert_eq!(field.square(&x_plus_1), field.mul(&x_plus_1, &x_plus_1));

// Addition is XOR; x · (x² + 1) = x³ + x ≡ 1, so they are inverses.
assert_eq!(
    Gf2m::add(&BigUint::from_u64(0b110), &BigUint::from_u64(0b011)),
    BigUint::from_u64(0b101)
);
let x = BigUint::from_u64(0b010);
assert_eq!(field.inverse(&x), Some(x2_plus_1));
assert_eq!(field.inverse(&BigUint::zero()), None);

// Tr(x) = 0 in GF(2³), so the half-trace solves z² + z = x.
let z = field.half_trace(&x);
assert_eq!(Gf2m::add(&field.square(&z), &z), x);

// The quadratic z² + z = c has a solution exactly when Tr(c) = 0; the
// half-trace produces it. Tr(x) = 0 and Tr(1) = 1 in GF(2³).
assert_eq!(field.trace(&x), 0);
assert_eq!(field.trace(&BigUint::one()), 1);

// Squaring is a bijection; sqrt inverts it. x is the unique root of x².
assert_eq!(field.sqrt(&BigUint::from_u64(0b100)), x);

// x generates the order-7 multiplicative group.
assert_eq!(field.pow(&x, &BigUint::from_u64(7)), BigUint::one());

// Division: x² / x = x; dividing by zero is refused.
assert_eq!(field.div(&BigUint::from_u64(0b100), &x), Some(x.clone()));
assert_eq!(field.div(&x, &BigUint::zero()), None);

// solve_quadratic is total across degrees — here in the AES byte field
// GF(2⁸), where the degree is even and the half-trace does not apply.
// c = a² + a guarantees solvability; the two roots are a and a + 1.
let aes = Gf2m::new(BigUint::from_u64(0x11B)).expect("the AES byte field");
let a = BigUint::from_u64(0x53);
let c = Gf2m::add(&aes.square(&a), &a);
let z = aes.solve_quadratic(&c).expect("solvable by construction");
assert_eq!(Gf2m::add(&aes.square(&z), &z), c);
assert!(z == a || z == Gf2m::add(&a, &BigUint::one()));

// Rabin's test guards the constructor's irreducibility contract:
// x³ + x + 1 passes, x³ + 1 = (x + 1)(x² + x + 1) fails, and the AES
// polynomial x⁸ + x⁴ + x³ + x + 1 passes.
assert!(Gf2m::is_irreducible(&BigUint::from_u64(0b1011)));
assert!(!Gf2m::is_irreducible(&BigUint::from_u64(0b1001)));
assert!(Gf2m::is_irreducible(&BigUint::from_u64(0x11B)));
```

## Number theory

### Divisibility

`gcd` and `lcm` by Euclid — `gcd`, `gcd_extended`, and `mod_inverse` share a
Lehmer-accelerated engine, and `gcd` switches to subquadratic Half-GCD above
~131 kbit; `gcd_extended` returns the Bézout triple `(g, s, t)` with
`g = a·s + b·t`. `gcd_u64` is the word-sized form — single-word Euclid, the
base case the wide `gcd` falls to, public so callers holding machine words
(sieve coordinates, residues, small cofactors) skip the heap entirely.

```rust
let a = BigUint::from_u64(240);
let b = BigUint::from_u64(46);

assert_eq!(gcd(&a, &b), BigUint::from_u64(2));
assert_eq!(
    lcm(&BigUint::from_u64(4), &BigUint::from_u64(6)),
    BigUint::from_u64(12)
);

let (g, s, t) = gcd_extended(&a, &b);
let bezout = s.mul_biguint(&a).add(&t.mul_biguint(&b));
assert_eq!(bezout, BigInt::from_biguint(g));

// The word-sized form answers without an allocation.
assert_eq!(gcd_u64(240, 46), 2);
assert_eq!(gcd_u64(0, 7), 7); // gcd(0, b) = b
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
(`None` when the gcd exceeds one; `mod_inverse_u64` is its word-sized
companion), `mod_sqrt` by Tonelli–Shanks with a
dispatch to Cipolla's algorithm where the prime's 2-adic depth makes the
descent quadratic (`None` for non-residues; the result is verified by
squaring, so a composite modulus also yields `None`). `crt_combine` performs
ordered Chinese remaindering; `crt_combine_balanced` gives the same canonical
answer through balanced partial products and bounded parallel workers. Both
return `None` when the moduli are empty, zero, or not pairwise coprime. The
balanced form takes a maximum worker count, caps it at reported machine
parallelism, and treats zero as an explicit serial request.

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

// The word-sized companion: extended Euclid with i128 cofactors, so it is
// total over u64; panics only on a zero modulus.
assert_eq!(mod_inverse_u64(3, 7), Some(5));
assert_eq!(mod_inverse_u64(2, 4), None); // shares a factor

let root = mod_sqrt(&BigUint::from_u64(2), &p).expect("2 is a residue mod 41");
assert_eq!(BigUint::mod_mul(&root, &root, &p), BigUint::from_u64(2));
assert_eq!(mod_sqrt(&BigUint::from_u64(3), &p), None); // non-residue

// Sunzi's classic: 2 mod 3, 3 mod 5, 2 mod 7.
let x = crt_combine(&[
    (BigUint::from_u64(2), BigUint::from_u64(3)),
    (BigUint::from_u64(3), BigUint::from_u64(5)),
    (BigUint::from_u64(2), BigUint::from_u64(7)),
])
.expect("moduli are pairwise coprime");
assert_eq!(x, BigUint::from_u64(23));
assert_eq!(
    crt_combine_balanced(
        &[
            (BigUint::from_u64(2), BigUint::from_u64(3)),
            (BigUint::from_u64(3), BigUint::from_u64(5)),
            (BigUint::from_u64(2), BigUint::from_u64(7)),
        ],
        4,
    ),
    Some(x)
);
```

### Batch inversion

`mod_inverse_batch` inverts a whole slice for one inversion and
`3(n − 1)` multiplications — Montgomery's trick. `None` when any element
shares a factor with the modulus (which element is not identified — that
would cost the inversions the trick avoids).

```rust
let m = BigUint::from_u64(97);
let values = [BigUint::from_u64(3), BigUint::from_u64(10), BigUint::from_u64(96)];
let inverses = mod_inverse_batch(&values, &m).expect("all coprime to 97");
for (inv, v) in inverses.iter().zip(&values) {
    assert!(BigUint::mod_mul(inv, v, &m).is_one());
}
assert_eq!(mod_inverse_batch(&[BigUint::from_u64(0)], &m), None);
```

### Valuation

`valuation(n, p)` is the exponent of `p` in `n`; `remove_factor(n, p)`
returns the cofactor with the exponent (the shape of GMP's `mpz_remove`),
dividing through a ladder of squared powers so large valuations cost
logarithmically many divisions. Both panic on `n = 0` (unbounded) or
`p < 2`.

```rust
let n = BigUint::from_u64(3_888); // 2^4 · 3^5
assert_eq!(valuation(&n, &BigUint::from_u64(2)), 4);
let (cofactor, exponent) = remove_factor(&n, &BigUint::from_u64(3));
assert_eq!(exponent, 5);
assert_eq!(cofactor, BigUint::from_u64(16));
```

### Rational reconstruction

`rational_reconstruct` recovers the unique fraction `p/q` from its image
`x ≡ p·q⁻¹ (mod m)`, with `|p|` and `q` at most `⌊√((m−1)/2)⌋`;
`rational_reconstruct_bounded` takes explicit bounds `N`, `D` and panics
unless `2·N·D < m` — the caller's contract, which is what makes the
answer unique. `None` means no fraction within the bounds reduces to `x`.
This is the recovery step of CRT-lifted and p-adic computation: compute
with residues, then read the rational answer back. When reconstructing
many values under one modulus, compute the bound once and use the bounded
form — the symmetric wrapper recomputes a square root that costs more
than the reconstruction itself at large sizes.

```rust
// 22/7 survives reduction mod 1009 and comes back intact.
let m = BigUint::from_u64(1_009);
let seven_inv = mod_inverse(&BigUint::from_u64(7), &m).expect("7 is invertible");
let x = BigUint::mod_mul(&BigUint::from_u64(22), &seven_inv, &m);
let (p, q) = rational_reconstruct(&x, &m).expect("22/7 is within the bounds");
assert_eq!(p, BigInt::from_biguint(BigUint::from_u64(22)));
assert_eq!(q, BigUint::from_u64(7));

// Negative numerators carry their sign: −3/5 mod 1009.
let five_inv = mod_inverse(&BigUint::from_u64(5), &m).expect("5 is invertible");
let x = m.sub(&BigUint::mod_mul(&BigUint::from_u64(3), &five_inv, &m));
let (p, q) = rational_reconstruct(&x, &m).expect("-3/5 is within the bounds");
assert_eq!(p, BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)));
assert_eq!(q, BigUint::from_u64(5));

// The bounded form is explicit about its contract.
assert_eq!(
    rational_reconstruct_bounded(&x, &m, &BigUint::from_u64(3), &BigUint::from_u64(5)),
    Some((
        BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)),
        BigUint::from_u64(5)
    ))
);
```

### Bulk primes, word division, and size estimates

`primes_below(bound)` sieves every prime below `bound` into a `Vec<u64>` —
the bulk companion to `is_probable_prime`. `div_rem_u64` divides by a
machine word returning quotient and remainder without a heap-allocated
divisor, and `to_u64` narrows to a word when the value fits (`None`
otherwise), the checked counterpart to `low_u128`. `to_f64_lossy`
saturates to infinity above the `f64` range and `ln_approx` stays finite
far past it, for size-driven parameter heuristics. `digit_count(radix)`
answers "how long is this number written in that radix" from the limbs —
a logarithm with the power-of-radix boundary settled by comparison —
without producing the expansion that `to_str_radix(radix).len()` would
build only to throw away; zero has one digit, and a radix below two
panics.

```rust
assert_eq!(primes_below(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);

let n = BigUint::from_u64(1_000);
assert_eq!(n.div_rem_u64(7), (BigUint::from_u64(142), 6));
assert_eq!(n.to_u64(), Some(1_000));
assert_eq!(BigUint::from_u128(1u128 << 64).to_u64(), None);

assert_eq!(BigUint::from_u64(8).to_f64_lossy(), 8.0);
assert!((BigUint::from_u64(1_000).ln_approx() - 1_000f64.ln()).abs() < 1e-9);

// Digit counts without the digits: the boundary cases are the powers.
assert_eq!(BigUint::from_u64(1_000).digit_count(10), 4);
assert_eq!(BigUint::from_u64(999).digit_count(10), 3);
assert_eq!(BigUint::from_u64(255).digit_count(16), 2); // "ff"
assert_eq!(BigUint::zero().digit_count(10), 1); // written "0"
```

### Division by a divisor that does not change

`WordReciprocal::new(d)` precomputes the reciprocal of a `u64` divisor once, so
that later divisions by `d` cost a multiplication and a correction instead of
a hardware divide (Möller & Granlund, IEEE ToC 60 (2011), Algorithm 4). It
answers exactly what `div_rem_u64` and `rem_u64` answer; the difference is
only where the cost sits. Build one per divisor and keep it — a table built
once and consulted for a whole run is the shape it is for. Construction panics
on a zero divisor.

`rem_euclid_i64` is the signed entry point, returning the *non-negative*
residue in `0..d`, which is what an index into a table wants and what a
truncating `%` does not give.

```rust
// Non-zero is the whole precondition, so the type carries it: there is no
// error case left for the caller to handle.
let r = WordReciprocal::new(NonZeroU64::new(1_000_003).expect("literal is non-zero"));
assert_eq!(r.divisor(), 1_000_003);

// The same answers as the hardware-division path. Note the argument roles:
// `BigUint::rem_u64` takes the divisor, `WordReciprocal::rem` takes the dividend.
assert_eq!(r.div_rem(2_000_007), (2, 1));
assert_eq!(r.rem(2_000_006), 0);

// Non-negative residues for signed positions.
assert_eq!(r.rem_euclid_i64(1_000_005), 2);
assert_eq!(r.rem_euclid_i64(-1), 1_000_002);
assert_eq!(r.rem_euclid_i64(-1_000_003), 0);

// Multi-limb dividends go through the same kernel, and are where it wins.
let n = BigUint::from_u128(340_282_366_920_938_463_463_374_607_431_768_211_455);
assert_eq!(n.rem_reciprocal(&r), n.rem_u64(1_000_003));
assert_eq!(n.div_rem_reciprocal(&r).0, n.div_rem_u64(1_000_003).0);
```

### Prime-power square roots

`mod_sqrt_prime_power(a, p, e)` returns every square root of `a` rem
`p^e`, ascending, empty for a non-residue — Hensel's lift for odd `p`, the
mod-8 structure for `p = 2`, and the valuation reduction for `a` divisible
by `p`.

```rust
// 9 has four square roots mod 16: 3, 5, 11, 13.
let roots = mod_sqrt_prime_power(&BigUint::from_u64(9), &BigUint::from_u64(2), 4);
assert_eq!(roots, vec![
    BigUint::from_u64(3), BigUint::from_u64(5),
    BigUint::from_u64(11), BigUint::from_u64(13),
]);
```

### Batch smoothness

`smooth_parts(values, primes)` returns the smooth part of each value over
the prime set — the largest divisor built only from those primes, with
multiplicity — in one batched pass. A value is fully smooth when its smooth
part equals itself; a zero value maps to zero and a one to one.

```rust
let primes = primes_below(10); // 2, 3, 5, 7
let values = [
    BigUint::from_u64(360),  // 2^3 · 3^2 · 5 — fully smooth
    BigUint::from_u64(2 * 11), // 11 is not in the base
];
let parts = smooth_parts(&values, &primes);
assert_eq!(parts[0], BigUint::from_u64(360));
assert_eq!(parts[1], BigUint::from_u64(2));
```

`SmoothnessBase::new(primes)` is the same algorithm with the primes' product
built once and kept, for a caller that wants to choose its own batch size.
The free function above rebuilds that product on every call, which for a base
of a few thousand primes is a product of tens of thousands of bits: paid once
per run it is nothing, but paid per batch it dictates how the caller may
batch. The context lifts that constraint — batches may be as small as the
caller likes, and the answers do not depend on how they are grouped. The
obligation that every entry is at least two is checked at construction rather
than per batch.

```rust
let base = SmoothnessBase::new(&primes_below(10)).expect("every entry is at least two");
assert_eq!(base.primes(), &[2, 3, 5, 7]);

// The same answers as the free function, in batches of the caller's choosing.
let batch = base.smooth_parts(&[BigUint::from_u64(360), BigUint::from_u64(2 * 11)]);
assert_eq!(batch[0], BigUint::from_u64(360));
assert_eq!(batch[1], BigUint::from_u64(2));
assert_eq!(base.smooth_parts(&[BigUint::from_u64(360)])[0], batch[0]);

// Smoothness is the predicate: the smooth part equals the value.
assert!(batch[0] == BigUint::from_u64(360));
assert!(batch[1] != BigUint::from_u64(2 * 11));
```

The batch is built from `product_tree` (the values' pairwise-product tree)
and `remainder_tree` (a modulus reduced against every leaf in one descent).
`product_tree` returns a `ProductTree`, whose levels are private: the
descent is correct only for the exact shape the constructor produces, so
construction is the only way to obtain one. Read it back with `root()`,
`leaves()`, `len()`, `is_empty()`, or `levels()` for the intermediate
products. `remainder_tree` panics on a zero leaf, which it would divide by.

```rust
let values = [BigUint::from_u64(7), BigUint::from_u64(11), BigUint::from_u64(13)];
let tree = product_tree(&values);
assert_eq!(*tree.root().unwrap(), BigUint::from_u64(7 * 11 * 13)); // 1001
assert_eq!(tree.len(), 3);

let residues = remainder_tree(&tree, &BigUint::from_u64(100));
assert_eq!(residues, vec![
    BigUint::from_u64(100 % 7),
    BigUint::from_u64(100 % 11),
    BigUint::from_u64(100 % 13),
]);
```

### Primality

`is_probable_prime` runs trial division plus Miller-Rabin over the twelve
fixed small-prime bases; `miller_rabin_with_bases` takes an explicit
base set (after the same unconditional small-prime sieve, so it is a
probable-prime predicate, not a way to run Miller-Rabin with exactly those
bases and nothing else); `miller_rabin_witness` is the single-round
primitive for callers with their own witness schedule. Each base it takes
is reduced rem the candidate and the trivial residues `{0, 1, n−1}` are
discarded; a set with no effective round proves nothing and is not prime.
The fixed bases are for candidates you generated yourself — an adversary
can construct pseudoprimes against any fixed base set, so untrusted input
needs extra candidate-derived witnesses (as the parent cryptography crate's
`is_probable_prime_untrusted` adds).

```rust
assert!(is_probable_prime(&BigUint::from_u64(65_537)));
assert!(!is_probable_prime(&BigUint::from_u64(561))); // Carmichael

// 2047 = 23 · 89 is the smallest strong pseudoprime to base 2: the base-2
// Miller-Rabin round is fooled (2 is not a witness), but base 3 exposes it.
let n = BigUint::from_u64(2_047);
assert!(!miller_rabin_witness(&n, &BigUint::from_u64(2))); // 2 is not a witness
assert!(miller_rabin_witness(&n, &BigUint::from_u64(3))); // 3 is
assert!(!is_probable_prime(&n)); // the full test rejects it

// with_bases reduces each base rem n and discards {0, 1, n-1}; a set of
// only trivial bases runs no effective round and is not reported prime.
let composite = BigUint::from_u64(1_022_117); // 1009 × 1013, survives the sieve
assert!(!miller_rabin_with_bases(&composite, &[1])); // 1 never testifies
assert!(!miller_rabin_with_bases(&composite, &[2])); // 2 exposes it
```

`is_probable_prime_bpsw` is the Baillie–PSW test: trial division, one
strong base-2 Miller–Rabin round, then the strong Lucas test with
Selfridge's parameters (`is_strong_lucas_probable_prime` exposes that
stage alone). The two stages fail on disjoint kinds of composites as far
as anyone has found — no composite passing both is known, and none exists
below 2⁶⁴, where the test is therefore deterministic.

```rust
assert!(is_probable_prime_bpsw(&BigUint::from_u64(65_537)));

// 2047 fools base 2; the Lucas stage rejects it.
assert!(!is_probable_prime_bpsw(&BigUint::from_u64(2_047)));

// 5459 is a strong Lucas pseudoprime; the base-2 stage rejects it.
assert!(is_strong_lucas_probable_prime(&BigUint::from_u64(5_459)));
assert!(!is_probable_prime_bpsw(&BigUint::from_u64(5_459)));
```

`is_prime_aks` is the exact deterministic Agrawal–Kayal–Saxena test. It
proves its answer unconditionally by checking polynomial congruences in
`(ℤ/nℤ)[X]/(X^r − 1)` after the perfect-power, multiplicative-order, and
small-gcd stages. Its polynomial running-time result is theoretically
important, but the constants are large: use the Miller–Rabin or Baillie–PSW
interfaces for practical probable-prime testing, and call AKS when this exact
algorithm is specifically required. A `true` AKS result is an unconditional
primality proof, not a probable-prime verdict; `false` means composite.

```rust
// Unlike the probable-prime predicates, AKS is an unconditional proof.
assert!(is_prime_aks(&BigUint::from_u64(101)));
assert!(!is_prime_aks(&BigUint::from_u64(561))); // Carmichael
```

## Polynomials

`PolyZ` is a dense univariate polynomial over ℤ and `PolyMod` one over
ℤ/mℤ; both store coefficients low-to-high and normalize away trailing
zeros, so the zero polynomial is the empty coefficient list. `PolyZ`
offers `add`/`sub`/`mul`, `evaluate` (Horner), `derivative`, `content` and
`primitive_part`, `div_rem` (exact division over ℤ, `None` when it does not
divide evenly), and `pseudo_div_rem` (the always-defined integer-preserving
form). `PolyMod` adds `div_rem`/`rem`/`gcd`/`make_monic`/
`mod_pow` — the division-based operations invert a leading coefficient, so
they require a prime modulus and panic on a non-invertible pivot.

```rust
// (x + 1)(x + 2) = x^2 + 3x + 2, over ℤ.
let a = PolyZ::from_i64_slice(&[1, 1]); // x + 1
let b = PolyZ::from_i64_slice(&[2, 1]); // x + 2
assert_eq!(a.mul(&b), PolyZ::from_i64_slice(&[2, 3, 1]));

// Exact division over ℤ: (x^2 + 3x + 2) / (x + 1) = x + 2, no remainder.
let (q, r) = a.mul(&b).div_rem(&a).expect("x + 1 divides");
assert_eq!(q, b); // x + 2
assert!(r.is_zero());
// A leading coefficient that does not divide has no integer quotient.
assert_eq!(
    PolyZ::from_i64_slice(&[1, 0, 1]).div_rem(&PolyZ::from_i64_slice(&[1, 2])),
    None
);

// Evaluation and derivative.
assert_eq!(a.mul(&b).evaluate(&BigInt::from_i64(3)), BigInt::from_i64(20)); // 9+9+2
assert_eq!(a.mul(&b).derivative(), PolyZ::from_i64_slice(&[3, 2])); // 2x + 3

// content and primitive part of 6x^2 + 9x + 15.
let c = PolyZ::from_i64_slice(&[15, 9, 6]);
assert_eq!(c.content(), BigInt::from_i64(3));
assert_eq!(c.primitive_part(), PolyZ::from_i64_slice(&[5, 3, 2]));

// Over ℤ/7ℤ: (x - 1)(x - 2) and (x - 2)(x - 3) share x - 2.
let p = BigUint::from_u64(7);
let f = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[2, -3, 1]), &p); // x^2 - 3x + 2
let g = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[6, -5, 1]), &p); // x^2 - 5x + 6
let shared = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[-2, 1]), &p); // x - 2
assert_eq!(f.gcd(&g), shared);
```

`resultant` and `discriminant` characterize shared and repeated factors:
the resultant is zero exactly when two polynomials share a non-constant
factor, the discriminant zero exactly when one has a repeated factor. Both
are computed by fraction-free (Bareiss) determinant of the Sylvester
matrix, staying in ℤ.

```rust
// disc(x^2 + 5x + 6) = 25 - 24 = 1.
let quad = PolyZ::from_i64_slice(&[6, 5, 1]);
assert_eq!(quad.discriminant(), BigInt::from_i64(1));

// (x-1)(x-2) and (x-1)(x-3) share (x-1): resultant zero.
let u = PolyZ::from_i64_slice(&[2, -3, 1]); // x^2 - 3x + 2
let v = PolyZ::from_i64_slice(&[3, -4, 1]); // x^2 - 4x + 3
assert_eq!(u.resultant(&v), BigInt::zero());

// A perfect square has discriminant zero: (x-1)^2 = x^2 - 2x + 1.
assert_eq!(PolyZ::from_i64_slice(&[1, -2, 1]).discriminant(), BigInt::zero());
```

Over a prime modulus, `PolyMod` factors and finds roots: `factor` returns
monic irreducibles with multiplicities, `is_irreducible` tests a single
polynomial, and `roots` gives the residues where it vanishes. Factoring and
root-finding are randomized (Cantor–Zassenhaus), so they take an `RandomSource` (see
Random sampling below for the trait; a minimal one is used here).

```rust
struct Lcg(u64);
impl RandomSource for Lcg {
    fn fill_bytes(&mut self, d: &mut [u8]) {
        for b in d {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (self.0 >> 33) as u8;
        }
    }
}
let p = BigUint::from_u64(7);
let mut rng = Lcg(0x1234_5678);

// x^2 - 1 = (x - 1)(x + 1) mod 7 — two roots.
let f = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[-1, 0, 1]), &p);
let mut roots = f.roots(&mut rng);
roots.sort();
assert_eq!(roots, vec![BigUint::from_u64(1), BigUint::from_u64(6)]); // 6 ≡ -1

// x^2 + 1 is irreducible mod 7 (−1 is a non-residue), so no roots.
let g = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
assert!(g.is_irreducible());
assert!(g.roots(&mut rng).is_empty());

// factor returns monic irreducibles with multiplicities:
// x^3 + x = x · (x^2 + 1) over F_7.
let h = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[0, 1, 0, 1]), &p);
let mut fs = h.factor(&mut rng);
fs.sort_by_key(|(fac, _)| fac.degree());
assert_eq!(fs.len(), 2);
assert_eq!(fs[0].0.degree(), Some(1)); // x
assert_eq!(fs[1].0.degree(), Some(2)); // x^2 + 1
assert!(fs.iter().all(|(_, e)| *e == 1));
```

A further group serves computation in the quotient ring ℤ[x]/(f) with `f`
monic, and the traffic between ℤ and ℤ/mℤ that such computation involves.

`rem_monic` reduces rem a monic divisor without forming the quotient.
Division by a monic polynomial cannot fail, so unlike `div_rem` it returns
a value rather than an `Option`, and each step is one multiply-subtract
with no coefficient division at all. `product_mod_monic` multiplies many
polynomials in ℤ[x]/(f) by a product tree, reducing at every level:
pairing keeps both operands the same size, which is the shape `mul`'s
Karatsuba wants, and because reduction is a ring homomorphism the answer
equals the fold multiplied out and reduced once — at a fraction of the
cost, since no intermediate is ever allowed to exceed `deg f`.

`PolyMod::symmetric_lift` returns the ℤ[x] representative with
coefficients in (−m/2, m/2]. That is the lift to use when a modular
computation was meant to recover an integer answer: a modulus wider than
twice the height returns the true polynomial exactly, signs and all.
`PolyMod::change_modulus` re-reads the same coefficient representatives at
a different modulus. Narrowing (the new modulus divides the old) is the
ring projection and preserves sums and products; widening (the old divides
the new) is its canonical section, which preserves only equality — it is
the seeding step of a Newton lift, not a ring map.

`PolyZ::balanced_base_expansion` writes `n = Σ cₖ mᵏ` with every digit
below the top in the symmetric range, the representation the number-field
sieve's polynomial selection wants because it halves `max |cₖ|` at no
cost. `homogeneous_substitution` evaluates the homogenization
`F(X, Y) = Yᵈ f(X/Y)` at a pair of polynomials, which is how a norm
`bᵈ f(a/b)` or a change of coordinates is taken without leaving ℤ[x].
`roots_mod_prime_power` finds every root rem `pᵉ` by Hensel lifting
from the roots rem `p`, branching where the derivative vanishes.

```rust
struct Lcg2(u64);
impl RandomSource for Lcg2 {
    fn fill_bytes(&mut self, d: &mut [u8]) {
        for b in d {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (self.0 >> 33) as u8;
        }
    }
}
let mut rng = Lcg2(0x2024_1111);

// Reduction rem the monic x^2 + 1 is a ring homomorphism.
let f = PolyZ::from_i64_slice(&[1, 0, 1]);
let a = PolyZ::from_i64_slice(&[3, 2, 5]); // 5x^2 + 2x + 3
let b = PolyZ::from_i64_slice(&[1, 7]); // 7x + 1
assert_eq!(
    a.mul(&b).rem_monic(&f),
    a.rem_monic(&f).mul(&b.rem_monic(&f)).rem_monic(&f)
);

// The product tree in Z[x]/(f) agrees with the fold, reduced once.
let x = PolyZ::from_i64_slice(&[0, 1]);
let factors = [a.clone(), b.clone(), x.clone()];
assert_eq!(
    PolyZ::product_mod_monic(&factors, &f),
    a.mul(&b).mul(&x).rem_monic(&f)
);

// Balanced base-1000 expansion of 1234567890: digits in (-500, 500].
let n = BigInt::from_i64(1_234_567_890);
let base = BigUint::from_u64(1_000);
let g = PolyZ::balanced_base_expansion(&n, &base, 3);
assert_eq!(g, PolyZ::from_i64_slice(&[-110, -432, 235, 1]));
assert_eq!(g.evaluate(&BigInt::from_i64(1_000)), n); // exact, by construction

// Homogenization of x^2 + 1 at (2x, 3): 3^2·1 + (2x)^2 = 4x^2 + 9.
assert_eq!(
    f.homogeneous_substitution(
        &PolyZ::from_i64_slice(&[0, 2]),
        &PolyZ::from_i64_slice(&[3])
    ),
    PolyZ::from_i64_slice(&[9, 0, 4])
);

// Square roots of 2 rem 7^3 = 343, lifted from ±3 rem 7.
let sqrt2 = PolyZ::from_i64_slice(&[-2, 0, 1]).roots_mod_prime_power(
    &BigUint::from_u64(7),
    3,
    &mut rng,
);
assert_eq!(sqrt2, vec![BigUint::from_u64(108), BigUint::from_u64(235)]);
assert_eq!(108u64 * 108 % 343, 2);

// A symmetric lift recovers the integer polynomial it came from.
let wide = BigUint::from_u64(2).pow_u64(96);
assert_eq!(PolyMod::from_poly_z(&a, &wide).symmetric_lift(), a);
```

### Real roots

`PolyZ::real_roots` returns the real roots ascending, each repeated according
to its multiplicity. Multiplicities are settled exactly in ℤ first — by the
squarefree decomposition, from repeated gcd with the derivative — and floating
point is then used only to *locate* roots whose count is already known. That
matters because a bisection sees a root only where the polynomial changes
sign, so an even-multiplicity root is invisible to it, and two nearby simple
roots are indistinguishable from one double root at `f64` precision.

An empty `Ok` means no real roots, which is the ordinary answer for an
even-degree polynomial. The two refusals are distinct: `RealRootError::ZeroPolynomial`
(every real number is a root) and `RealRootError::CoefficientOutOfRange` (a
coefficient does not fit `f64`, so the polynomial cannot be evaluated at all).

```rust
// (x−1)(x−2)(x−3)
let f = PolyZ::from_i64_slice(&[-6, 11, -6, 1]);
let roots = f.real_roots().expect("finite coefficients");
assert_eq!(roots.len(), 3);
assert!((roots[0] - 1.0).abs() < 1e-9);
assert!((roots[2] - 3.0).abs() < 1e-9);

// (x−2)²: a double root, returned twice — bisection alone cannot see it.
let squared = PolyZ::from_i64_slice(&[4, -4, 1]);
assert_eq!(squared.real_roots().expect("finite").len(), 2);

// No real roots is an answer, not a failure.
assert_eq!(PolyZ::from_i64_slice(&[1, 0, 1]).real_roots(), Ok(Vec::new()));

// The zero polynomial is a refusal.
assert_eq!(PolyZ::zero().real_roots(), Err(RealRootError::ZeroPolynomial));
```

## Lattice reduction

`lll_reduce` applies the Lenstra–Lenstra–Lovász algorithm to an ordered
lattice basis — the rows of a `&mut [Vec<BigInt>]` — replacing it in place
with a short, nearly orthogonal basis of the same lattice. It is exact
(integer Gram–Schmidt, Cohen's Algorithm 2.6.3), never rational or
floating-point. The Lovász parameter defaults to `δ = 3/4`;
`lll_reduce_delta` takes it as a fraction in `(1/4, 1)`, larger reducing
more aggressively. The rows must be linearly independent.

```rust
let row = |xs: &[i64]| xs.iter().map(|&x| BigInt::from_i64(x)).collect::<Vec<_>>();
// A badly skewed basis for a 2-D lattice.
let mut basis = vec![row(&[201, 37]), row(&[1648, 297])];
lll_reduce(&mut basis);
// Reduction returns short vectors spanning the same lattice.
assert_eq!(basis, vec![row(&[1, 32]), row(&[40, 1])]);
```

`gauss_reduce_weighted` is the two-dimensional case, in machine integers and
under an anisotropic metric. In two dimensions Lagrange–Gauss reduction is
not a heuristic as LLL is in general dimension: it returns a shortest
non-zero vector of the lattice outright, first, with the second shortest
independent of it after. The metric is the diagonal form
`‖(x, y)‖² = (w₀·x)² + (w₁·y)²`.

For a skewed metric `(x/√s)² + (y·√s)²` with **rational** `s = p/q`, multiply
the form through by `pq` — a positive scale factor changes no comparison and
no rounding — which clears the square roots and gives `(q·x)² + (p·y)²`, so
pass `weights = [q, p]`. An integer skew is the case `q = 1`. A skew that is
not rational must be approximated first, and the reduction is then exact under
the approximating form rather than the intended one; the approximation is the
caller's choice, and it costs range, since the norms grow as the square of the
weights. It panics on a dependent pair, a non-positive weight, or arithmetic
past `i128` — and because the rounding step needs twice the norm, the working
bound is that norms fit `2¹²⁶`.

```rust
// Weights are NonZeroU64, so a non-positive weight cannot be written.
let nz = |n: u64| NonZeroU64::new(n).expect("literal is non-zero");

// A skew-12 lattice, reduced under the matching diagonal form.
let basis = [[1024i128, 0], [37, 1]];
let reduced = gauss_reduce_weighted(basis, [nz(1), nz(12)]).expect("a valid basis");
// Same lattice: the determinant is preserved up to sign.
let det = |b: [[i128; 2]; 2]| b[0][0] * b[1][1] - b[0][1] * b[1][0];
assert_eq!(det(reduced).abs(), det(basis).abs());

// Weights reorder what counts as short: under a heavy x-weight the
// y-axis vector wins, and under a heavy y-weight the x-axis vector does.
let square = [[1i128, 0], [0, 1]];
assert_eq!(gauss_reduce_weighted(square, [nz(100), nz(1)]), Ok([[0, 1], [1, 0]]));
assert_eq!(gauss_reduce_weighted(square, [nz(1), nz(100)]), Ok([[1, 0], [0, 1]]));

// A dependent pair is a typed error rather than a panic.
assert_eq!(
    gauss_reduce_weighted([[2, 4], [1, 2]], [nz(1), nz(1)]),
    Err(ReductionError::DependentBasis)
);
```

## Random sampling

Implement `RandomSource` — one method, `fill_bytes` — and every sampler is driven by
it. rump chooses no entropy source: output quality is exactly source
quality, so cryptographic callers must supply a CSPRNG. `random_below`
draws uniformly in `[0, upper)` by rejection, `random_nonzero_below` in
`[1, upper)`, `random_coprime_below` additionally requires coprimality, and
`random_probable_prime` searches for a prime of exactly the requested bit
length.

Each sampler carries a stall guard that **panics** on the degenerate
generators it can soundly detect, rather than looping forever; every guard
is sized so a working generator trips it with probability at most
`e⁻¹¹¹ ≈ 2⁻¹⁶⁰` (each function's rustdoc gives its own, tighter bound).
`random_below` and `random_nonzero_below` bound consecutive rejections
(acceptance is at least 1/2 per draw regardless of arguments);
`random_probable_prime` bounds fruitless rounds at `64·bits` (the cap
scales with the width, so it is sound at every width) and additionally
fails fast on a *pinned* generator by skipping the re-screen of a repeated
candidate. `random_coprime_below` is the one sampler where no usable
rejection count exists — legitimate arguments can make units arbitrarily
sparse — so its guard detects only a pinned generator; a degenerate source
cycling among several rejected values hangs it and remains the caller's to
avoid.

```rust
/// xorshift64: deterministic and compact. Fine for a manual; NOT a CSPRNG.
struct XorShift64(u64);

impl RandomSource for XorShift64 {
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
| `sub` / `-=` | the result would be negative |
| `div_rem` / `div_rem_u64` / `rem` / `rem_u64` | the divisor or modulus is zero |
| `BigUint::mod_mul` / `mod_pow` / `mod_add` / `mod_sub` | the modulus is zero |
| `ln_approx` | the value is zero |
| `digit_count` | the radix is below 2 |
| `mod_inverse_u64` | the modulus is zero |
| `nth_root_floor` | `k == 0` |
| `mod_sqrt_prime_power` | `e == 0` or `p < 2` |
| `valuation` / `remove_factor` | `n == 0` or `p < 2` |
| `rational_reconstruct_bounded` | `2·N·D >= m` (the uniqueness contract) |
| `from_str_radix` / `to_str_radix` | the radix is outside `2..=36` |
| `remainder_tree` / `smooth_parts` | a value (leaf) is zero — but `smooth_parts` maps a zero value to zero rather than reducing it |
| `Gf2m::half_trace` | the field degree is even |
| `Gf2m::trace` | a reducible modulus makes the Frobenius sum leave GF(2) |
| `PolyZ` / `PolyMod` division | the divisor is the zero polynomial |
| `PolyMod::new` / `zero` / `from_poly_z` | the modulus is below 2 |
| `PolyMod` (any two-operand op) | the operands carry different moduli |
| `PolyMod` division / `gcd` / `factor` | a non-invertible pivot (composite modulus) |
| `squarefree_factorization` / `factor` | the polynomial is constant or zero |
| `PolyMod::factor` / `roots` | the supplied `RandomSource` makes no progress (the equal-degree splitter's stall guard) |
| `PolyZ::rem_monic` / `product_mod_monic` | the divisor is zero or its leading coefficient is not 1 |
| `PolyZ::balanced_base_expansion` | the base is below 2 |
| `PolyZ::roots_mod_prime_power` | the exponent is zero, the base is below 2, the polynomial is zero or has every coefficient divisible by `pᵉ` (every residue is then a root), or the lift would exceed `MAX_ENUMERATED_ROOTS` candidates at some level or in its answer |
| `lll_reduce` / `lll_reduce_delta` | dependent, ragged, or zero-length rows; the `_delta` form also on `δ ∉ (1/4, 1)` or a zero denominator |
| `to_be_bytes_padded` | the value needs more than the requested byte length |
| `MontgomeryContext::mul_mont` / `square_mont` / their `_with_workspace` forms / `pow_encoded` | given an operand not reduced below the modulus — the shared in-domain contract, asserted in debug builds; in release a grossly over-width operand trips the internal bounds check. `encode` and `decode` instead reduce any representative and never panic on width |
| `random_below` / `random_nonzero_below` / `random_coprime_below` / `random_probable_prime` | the generator trips a stall guard — see Random sampling above for what each guard can and cannot detect |

Fallible mathematics — a missing inverse, a non-residue, an even Montgomery
modulus, non-coprime CRT moduli — returns `Option` instead.
