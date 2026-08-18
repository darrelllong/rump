//! The workload set. This is the only file that differs between the two
//! revision adapters, and only where the 0.3.0 rename forces it.
//!
//! v0.3.0 names used here: module paths (`rump::modular`, `rump::polynomial`,
//! `rump::finite_field`), `add`/`sub`/`mul`/`square` without the `_ref` suffix,
//! `rem` for `modulo`, `BarrettContext`/`MontgomeryContext`,
//! `to_residue`/`mul_residue`, `mod_sqrt`, `PolyMod`.

use crate::shared::{calibrate, small_primes, timed, Corpus, Digest, SplitMix};
use std::hint::black_box;

use rump::modular::{mod_inverse, mod_pow, mod_sqrt, BarrettContext, MontgomeryContext};
use rump::finite_field::Gf2m;
use rump::lattice::lll_reduce;
use rump::modular::mod_inverse_batch;
use rump::number_theory::{
    gcd, gcd_extended, jacobi, product_tree, remainder_tree, smooth_parts,
};
use rump::gf2::{block_lanczos_dependencies, dense_null_space, prune_singletons};
use rump::lattice::gauss_reduce_weighted;
use rump::integer::WordReciprocal;
use rump::number_theory::SmoothnessBase;
use rump::random::{random_below, random_coprime_below, random_probable_prime, RandomSource};
use std::num::NonZeroU64;
use rump::polynomial::PolyZ;
use rump::{BigInt, BigUint};

/// Fixed seed, so a case's draws are a property of the case and not of the run.
const SEED: u64 = 0x5eed_0fa4_d17d;

/// Bridges the shared generator to this revision's random-source trait.
struct Adapter(SplitMix);

impl RandomSource for Adapter {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill(dest);
    }
}

/// A `width`-bit integer drawn from the fixed seed, independent of any corpus
/// value. Deterministic, so both revisions reduce the same modulus.
fn independent_modulus(width: usize) -> BigUint {
    let mut draw = SplitMix(SEED);
    let digits = width.div_ceil(4).max(2);
    let mut hexits = String::with_capacity(digits);
    while hexits.len() < digits {
        hexits.push_str(&format!("{:016x}", draw.next_u64()));
    }
    hexits.truncate(digits);
    // Force the top hexit high so the value really is `width` bits wide, and
    // the last bit set so it is odd and shares no factor two with a leaf.
    let mut bytes: Vec<char> = hexits.chars().collect();
    bytes[0] = 'f';
    let last = bytes.len() - 1;
    bytes[last] = 'd';
    hex(&bytes.into_iter().collect::<String>())
}

/// Reads a packed GF(2) matrix: one row per line, space-separated hex words,
/// column `c` at bit `c % 64` of word `c / 64`.
fn gf2_matrix(c: &Corpus) -> (Vec<Vec<u64>>, usize) {
    let columns: usize = c
        .note("columns")
        .and_then(|v| v.parse().ok())
        .expect("gf2 matrix corpus declares a column count");
    let rows = c
        .items
        .iter()
        .map(|line| {
            line.split_whitespace()
                .map(|w| u64::from_str_radix(w, 16).expect("hex word"))
                .collect()
        })
        .collect();
    (rows, columns)
}

/// Word-sized divisors and dividends from a corpus of wide operands: the low
/// 64 bits of each. Divisors are forced nonzero, since the reciprocal is only
/// defined for a nonzero divisor.
fn word_pairs(c: &Corpus) -> (Vec<NonZeroU64>, Vec<u64>) {
    let mut divisors = Vec::new();
    let mut values = Vec::new();
    for pair in c.items.chunks(2) {
        if pair.len() < 2 {
            break;
        }
        let low = |s: &String| {
            let tail = &s[s.len().saturating_sub(16)..];
            u64::from_str_radix(tail, 16).unwrap_or(1)
        };
        divisors.push(NonZeroU64::new(low(&pair[0])).unwrap_or(NonZeroU64::MIN));
        values.push(low(&pair[1]));
    }
    (divisors, values)
}

/// A built workload: operands already parsed, ready to time.
pub struct Work {
    /// Operations per traversal, for the ns/op conversion.
    ops: usize,
    run: Box<dyn FnMut()>,
    /// Canonical result strings, produced outside the timed region.
    results: Vec<String>,
}

impl Work {
    pub fn digest(&self) -> String {
        let mut d = Digest::new();
        for r in &self.results {
            d.add(r);
        }
        d.finish()
    }
    pub fn results(&self) -> &[String] {
        &self.results
    }
    pub fn calibrate(&mut self) -> usize {
        calibrate(20.0, &mut *self.run)
    }
    pub fn time(&mut self, repeat: usize) -> f64 {
        let ops = self.ops;
        timed(repeat, ops, &mut *self.run)
    }
}

fn hex(s: &str) -> BigUint {
    BigUint::from_str_radix(s, 16).expect("corpus operand is hex")
}

/// Operands as (a, b) pairs.
fn pairs(c: &Corpus) -> Vec<(BigUint, BigUint)> {
    c.items
        .chunks(2)
        .filter(|ch| ch.len() == 2)
        .map(|ch| (hex(&ch[0]), hex(&ch[1])))
        .collect()
}

fn singles(c: &Corpus) -> Vec<BigUint> {
    c.items.iter().map(|s| hex(s)).collect()
}

/// Triples as (base, exponent, modulus).
fn triples(c: &Corpus) -> Vec<(BigUint, BigUint, BigUint)> {
    c.items
        .chunks(3)
        .filter(|ch| ch.len() == 3)
        .map(|ch| (hex(&ch[0]), hex(&ch[1]), hex(&ch[2])))
        .collect()
}

fn polys(c: &Corpus) -> Vec<PolyZ> {
    c.groups
        .iter()
        .map(|g| {
            PolyZ::new(
                g.iter()
                    .map(|s| BigInt::from_biguint(hex(s)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

pub fn build(case: &str, c: &Corpus) -> Work {
    match case {
        "int_add" => {
            let ops = pairs(c);
            let results = ops.iter().map(|(a, b)| a.add(b).to_str_radix(16)).collect();
            let data = ops;
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.add(b));
                    }
                }),
            }
        }
        "int_sub" => {
            let ops: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, b)| if a >= b { (a, b) } else { (b, a) })
                .collect();
            let results = ops.iter().map(|(a, b)| a.sub(b).to_str_radix(16)).collect();
            let data = ops;
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.sub(b));
                    }
                }),
            }
        }
        "int_mul" => {
            let data = pairs(c);
            let results = data.iter().map(|(a, b)| a.mul(b).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.mul(b));
                    }
                }),
            }
        }
        "int_square" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.square().to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.square());
                    }
                }),
            }
        }
        "int_div_rem" => {
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .filter(|(_, b)| !b.is_zero())
                .collect();
            let results = data
                .iter()
                .map(|(a, b)| {
                    let (q, r) = a.div_rem(b);
                    format!("{} {}", q.to_str_radix(16), r.to_str_radix(16))
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.div_rem(b));
                    }
                }),
            }
        }
        "int_to_hex" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.to_str_radix(16));
                    }
                }),
            }
        }
        "int_to_dec" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.to_str_radix(10)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.to_str_radix(10));
                    }
                }),
            }
        }
        "nt_gcd" => {
            let data = pairs(c);
            let results = data.iter().map(|(a, b)| gcd(a, b).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(gcd(a, b));
                    }
                }),
            }
        }
        "nt_gcd_extended" => {
            let data = pairs(c);
            let results = data
                .iter()
                .map(|(a, b)| {
                    let (g, _, _) = gcd_extended(a, b);
                    g.to_str_radix(16)
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(gcd_extended(a, b));
                    }
                }),
            }
        }
        "nt_jacobi" => {
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, b)| {
                    let mut n = b;
                    if !n.is_odd() {
                        n = n.add(&BigUint::one());
                    }
                    (a, n)
                })
                .collect();
            let results = data
                .iter()
                .map(|(a, n)| format!("{:?}", jacobi(a, n)))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, n) in &data {
                        black_box(jacobi(a, n));
                    }
                }),
            }
        }
        "mod_pow" => {
            let data = triples(c);
            let results = data
                .iter()
                .map(|(b, e, m)| mod_pow(b, e, m).to_str_radix(16))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (b, e, m) in &data {
                        black_box(mod_pow(b, e, m));
                    }
                }),
            }
        }
        "mod_inverse" => {
            let data = pairs(c);
            let results = data
                .iter()
                .map(|(a, m)| match mod_inverse(a, m) {
                    Some(i) => i.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, m) in &data {
                        black_box(mod_inverse(a, m));
                    }
                }),
            }
        }
        "mod_sqrt" => {
            let prime = hex(&c.items[0]);
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data
                .iter()
                .map(|a| match mod_sqrt(a, &prime) {
                    Some(r) => r.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(mod_sqrt(a, &prime));
                    }
                }),
            }
        }
        // Constructor cost, measured on its own.
        "barrett_new" => {
            let data: Vec<BigUint> = singles(c);
            let results = data
                .iter()
                .map(|m| {
                    BarrettContext::new(m)
                        .map(|ctx| ctx.modulus().to_str_radix(16))
                        .unwrap_or_else(|_| "err".to_string())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for m in &data {
                        black_box(BarrettContext::new(m).ok());
                    }
                }),
            }
        }
        "montgomery_new" => {
            let data: Vec<BigUint> = singles(c);
            let results = data
                .iter()
                .map(|m| {
                    MontgomeryContext::new(m)
                        .map(|ctx| ctx.modulus().to_str_radix(16))
                        .unwrap_or_else(|_| "err".to_string())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for m in &data {
                        black_box(MontgomeryContext::new(m).ok());
                    }
                }),
            }
        }
        // Reuse cost with a shared context: the setup is outside the timed
        // region, so this is the steady-state kernel.
        "barrett_mod_mul" => {
            let moduli: Vec<BigUint> = singles(c);
            let contexts: Vec<BarrettContext> = moduli
                .iter()
                .filter_map(|m| BarrettContext::new(m).ok())
                .collect();
            let operands: Vec<(BigUint, BigUint)> = moduli
                .iter()
                .map(|m| (m.sub(&BigUint::one()), m.sub(&BigUint::from_u64(2))))
                .collect();
            let results = contexts
                .iter()
                .zip(&operands)
                .map(|(ctx, (a, b))| ctx.mod_mul(a, b).to_str_radix(16))
                .collect();
            let data: Vec<(BarrettContext, BigUint, BigUint)> = contexts
                .into_iter()
                .zip(operands)
                .map(|(ctx, (a, b))| (ctx, a, b))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (ctx, a, b) in &data {
                        black_box(ctx.mod_mul(a, b));
                    }
                }),
            }
        }
        "montgomery_mul_residue" => {
            let moduli: Vec<BigUint> = singles(c);
            let mut prepared = Vec::new();
            let mut results = Vec::new();
            for m in &moduli {
                let Ok(ctx) = MontgomeryContext::new(m) else {
                    continue;
                };
                let a = ctx.to_residue(&m.sub(&BigUint::one()));
                let b = ctx.to_residue(&m.sub(&BigUint::from_u64(2)));
                let product = ctx.mul_residue(&a, &b).expect("same context");
                results.push(
                    ctx.from_residue(&product)
                        .expect("same context")
                        .to_str_radix(16),
                );
                prepared.push((ctx, a, b));
            }
            Work {
                ops: prepared.len(),
                results,
                run: Box::new(move || {
                    for (ctx, a, b) in &prepared {
                        black_box(ctx.mul_residue(a, b).expect("same context"));
                    }
                }),
            }
        }
        "poly_mul" => {
            let data: Vec<(PolyZ, PolyZ)> = polys(c)
                .chunks(2)
                .filter(|ch| ch.len() == 2)
                .map(|ch| (ch[0].clone(), ch[1].clone()))
                .collect();
            let results = data
                .iter()
                .map(|(f, g)| {
                    let p = f.mul(g);
                    format!("{:?}", p.degree())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (f, g) in &data {
                        black_box(f.mul(g));
                    }
                }),
            }
        }
        "gf2m_mul" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let elements: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let data: Vec<(BigUint, BigUint)> = elements
                .chunks(2)
                .filter(|ch| ch.len() == 2)
                .map(|ch| (ch[0].clone(), ch[1].clone()))
                .collect();
            let results = data
                .iter()
                .map(|(a, b)| field.mul(a, b).to_str_radix(16))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(field.mul(a, b));
                    }
                }),
            }
        }
        "gf2m_square" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data.iter().map(|a| field.square(a).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.square(a));
                    }
                }),
            }
        }
        "gf2m_inverse" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..]
                .iter()
                .map(|s| hex(s))
                .filter(|a| !a.is_zero())
                .collect();
            let results = data
                .iter()
                .map(|a| match field.inverse(a) {
                    Some(i) => i.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.inverse(a));
                    }
                }),
            }
        }
        "gf2m_sqrt" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data.iter().map(|a| field.sqrt(a).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.sqrt(a));
                    }
                }),
            }
        }
        "lattice_lll" => {
            // One basis per group; the reduction is in place, so each timed
            // traversal works on a fresh clone. Cloning is inside the loop
            // because the operation consumes its input, and the same clone cost
            // is charged to both revisions identically.
            let dimension: usize = c
                .note("dimension")
                .and_then(|d| d.parse().ok())
                .expect("lattice corpus declares a dimension");
            let bases: Vec<Vec<Vec<BigInt>>> = c
                .groups
                .iter()
                .map(|g| {
                    g.chunks(dimension)
                        .map(|row| {
                            row.iter()
                                .map(|s| BigInt::from_biguint(hex(s)))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let results = bases
                .iter()
                .map(|b| {
                    let mut work = b.clone();
                    lll_reduce(&mut work);
                    let mut out = String::new();
                    for row in &work {
                        for entry in row {
                            out.push_str(&entry.to_str_radix(16));
                            out.push(',');
                        }
                        out.push(';');
                    }
                    out
                })
                .collect();
            Work {
                ops: bases.len(),
                results,
                run: Box::new(move || {
                    for b in &bases {
                        let mut work = b.clone();
                        lll_reduce(&mut work);
                        black_box(&work);
                    }
                }),
            }
        }
        "batch_mod_inverse" => {
            let values: Vec<BigUint> = singles(c);
            // A modulus coprime to the batch: the largest value plus two, made
            // odd, so inversion succeeds for the whole batch.
            let modulus = values
                .iter()
                .max()
                .expect("non-empty batch")
                .add(&BigUint::from_u64(2));
            let results = match mod_inverse_batch(&values, &modulus) {
                Some(v) => v.iter().map(|i| i.to_str_radix(16)).collect(),
                None => vec!["none".to_string()],
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(mod_inverse_batch(&values, &modulus));
                }),
            }
        }
        "batch_smooth_parts" => {
            let values: Vec<BigUint> = singles(c);
            let primes = small_primes(512);
            let results = smooth_parts(&values, &primes)
                .iter()
                .map(|s| s.to_str_radix(16))
                .collect();
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(smooth_parts(&values, &primes));
                }),
            }
        }
        "rand_below" => {
            let bounds: Vec<BigUint> = singles(c);
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                bounds
                    .iter()
                    .map(|b| match random_below(&mut rng, b) {
                        Some(v) => v.to_str_radix(16),
                        None => "none".to_string(),
                    })
                    .collect()
            };
            Work {
                ops: bounds.len(),
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    for b in &bounds {
                        black_box(random_below(&mut rng, b));
                    }
                }),
            }
        }
        "rand_coprime_below" => {
            let bounds: Vec<BigUint> = singles(c);
            // 2·3·5·7·11·13. Only φ(30030)/30030 ≈ 0.19 of the residues are
            // coprime to it, so roughly four draws in five are rejected — the
            // rejection loop is what this case is here to measure.
            let coprime_to = BigUint::from_u64(30030);
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                bounds
                    .iter()
                    .map(|b| match random_coprime_below(&mut rng, b, &coprime_to) {
                        Some(v) => v.to_str_radix(16),
                        None => "none".to_string(),
                    })
                    .collect()
            };
            Work {
                ops: bounds.len(),
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    for b in &bounds {
                        black_box(random_coprime_below(&mut rng, b, &coprime_to));
                    }
                }),
            }
        }
        "rand_probable_prime" => {
            // The bit width comes from the corpus note, not from the operands:
            // this sampler takes a width, and the corpus is here to name it.
            let bits: usize = c
                .note("bits")
                .and_then(|b| b.parse().ok())
                .expect("bound corpus declares a width");
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                vec![match random_probable_prime(&mut rng, bits) {
                    Some(v) => v.to_str_radix(16),
                    None => "none".to_string(),
                }]
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    black_box(random_probable_prime(&mut rng, bits));
                }),
            }
        }
        "batch_product_tree" => {
            let values: Vec<BigUint> = singles(c);
            let tree = product_tree(&values);
            let results = match tree.root() {
                Some(root) => vec![root.to_str_radix(16)],
                None => vec!["empty".to_string()],
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(product_tree(&values));
                }),
            }
        }
        "batch_remainder_tree" => {
            let values: Vec<BigUint> = singles(c);
            let tree = product_tree(&values);
            // A modulus drawn independently of the batch, at the width of the
            // product. `root + k` for small `k` is what Bernstein's algorithm
            // is *not* used for: it makes every remainder equal to `k`, which
            // both trivialises the arithmetic and reduces the digest to a
            // restatement of the batch size.
            let width = tree.root().map_or(64, BigUint::bits);
            let modulus = independent_modulus(width);
            let results = remainder_tree(&tree, &modulus)
                .iter()
                .map(|r| r.to_str_radix(16))
                .collect();
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(remainder_tree(&tree, &modulus));
                }),
            }
        }
        // ---- v0.3.0-only cases -------------------------------------------
        //
        // These APIs do not exist in v0.2.2, so there is nothing to pair them
        // against. They are measured for an absolute baseline, and the paired
        // wrapper is not used on them.
        "gf2_dense_null_space" => {
            let (rows, columns) = gf2_matrix(c);
            let results = dense_null_space(&rows, columns)
                .iter()
                .map(|dep| {
                    let mut t = String::new();
                    for index in dep {
                        t.push_str(&index.to_string());
                        t.push(',');
                    }
                    t
                })
                .collect();
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(dense_null_space(&rows, columns));
                }),
            }
        }
        "gf2_prune_singletons" => {
            let (rows, columns) = gf2_matrix(c);
            let pruned = prune_singletons(&rows, columns);
            let results = vec![format!("{} {}", pruned.rows().len(), pruned.columns())];
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(prune_singletons(&rows, columns));
                }),
            }
        }
        "gf2_block_lanczos" => {
            let (rows, columns) = gf2_matrix(c);
            // Reseeded per traversal: the solver is randomised, and a fixed
            // seed is what makes its cost a property of the matrix rather
            // than of where the generator happened to be.
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                match block_lanczos_dependencies(&rows, columns, &mut rng) {
                    Some(d) => d
                        .iter()
                        .map(|dep| {
                            let mut t = String::new();
                            for index in dep {
                                t.push_str(&index.to_string());
                                t.push(',');
                            }
                            t
                        })
                        .collect(),
                    None => vec!["none".to_string()],
                }
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    black_box(block_lanczos_dependencies(&rows, columns, &mut rng));
                }),
            }
        }
        "poly_real_roots" => {
            let poly = PolyZ::new(
                c.items
                    .iter()
                    .map(|line| BigInt::from_str_radix(line, 10).expect("signed decimal"))
                    .collect(),
            );
            // The roots are f64, so the digest rounds them: printing full
            // precision would make the digest depend on the last bit of a
            // bisection, which is not the property being checked.
            let results = match poly.real_roots() {
                Ok(roots) => roots.iter().map(|r| format!("{r:.6}")).collect(),
                Err(e) => vec![format!("{e}")],
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(poly.real_roots().ok());
                }),
            }
        }
        "word_reciprocal_rem" => {
            let (divisors, values) = word_pairs(c);
            let reciprocals: Vec<WordReciprocal> =
                divisors.iter().map(|d| WordReciprocal::new(*d)).collect();
            let results = reciprocals
                .iter()
                .zip(values.iter())
                .map(|(r, v): (&WordReciprocal, &u64)| format!("{}", r.rem(*v)))
                .collect();
            Work {
                ops: values.len(),
                results,
                run: Box::new(move || {
                    for (r, v) in reciprocals.iter().zip(values.iter()) {
                        black_box(r.rem(*v));
                    }
                }),
            }
        }
        "word_reciprocal_div_rem" => {
            let (divisors, values) = word_pairs(c);
            let reciprocals: Vec<WordReciprocal> =
                divisors.iter().map(|d| WordReciprocal::new(*d)).collect();
            let results = reciprocals
                .iter()
                .zip(values.iter())
                .map(|(r, v): (&WordReciprocal, &u64)| {
                    let (q, m) = r.div_rem(*v);
                    format!("{q} {m}")
                })
                .collect();
            Work {
                ops: values.len(),
                results,
                run: Box::new(move || {
                    for (r, v) in reciprocals.iter().zip(values.iter()) {
                        black_box(r.div_rem(*v));
                    }
                }),
            }
        }
        "smoothness_base_new" => {
            let primes = small_primes(1024);
            let results = vec![match SmoothnessBase::new(&primes) {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("{e}"),
            }];
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(SmoothnessBase::new(&primes).ok());
                }),
            }
        }
        "lattice_gauss_weighted" => {
            // 2×2 bases only, which is what the weighted reduction takes.
            let weights = [
                NonZeroU64::new(1).expect("nonzero"),
                NonZeroU64::new(3).expect("nonzero"),
            ];
            let bases: Vec<[[i128; 2]; 2]> = c
                .groups
                .iter()
                .filter(|g| g.len() >= 4)
                .map(|g| {
                    let e = |i: usize| i128::from_str_radix(&g[i], 16).unwrap_or(1);
                    [[e(0), e(1)], [e(2), e(3)]]
                })
                .collect();
            let results = bases
                .iter()
                .map(|b| match gauss_reduce_weighted(*b, weights) {
                    Ok(r) => format!("{r:?}"),
                    Err(e) => format!("{e}"),
                })
                .collect();
            Work {
                ops: bases.len(),
                results,
                run: Box::new(move || {
                    for b in &bases {
                        black_box(gauss_reduce_weighted(*b, weights));
                    }
                }),
            }
        }
        "stream_add" => {
            // Operands are generated here, not read: see the note in
            // corpus-gen's `stream_corpus`. The working set is the point of
            // the case, so it is stated in the corpus header and checked
            // against what was actually built.
            let bits: usize = c.note("bits").and_then(|v| v.parse().ok()).expect("bits");
            let count: usize = c.note("count").and_then(|v| v.parse().ok()).expect("count");
            let seed: u64 = c
                .note("seed")
                .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                .expect("seed");
            let mut gen = SplitMix(seed);
            let mut wide = || {
                let mut n = BigUint::zero();
                // Set the top bit first so the buffer is sized once, then
                // fill the rest; growing it a bit at a time would dominate.
                n.set_bit(bits - 1);
                for index in 0..bits - 1 {
                    if gen.next_u64() & 1 == 1 {
                        n.set_bit(index);
                    }
                }
                n
            };
            let data: Vec<(BigUint, BigUint)> = (0..count).map(|_| (wide(), wide())).collect();
            let results = vec![format!(
                "{} {}",
                data.len(),
                data.iter().fold(0usize, |acc, (a, b)| acc ^ a.add(b).bits())
            )];
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.add(b));
                    }
                }),
            }
        }
        other => panic!("unknown case {other}"),
    }
}
