//! The workload set. This is the only file that differs between the two
//! revision adapters, and only where the 0.3.0 rename forces it.
//!
//! v0.3.0 names used here: module paths (`rump::modular`, `rump::polynomial`,
//! `rump::finite_field`), `add`/`sub`/`mul`/`square` without the `_ref` suffix,
//! `rem` for `modulo`, `BarrettContext`/`MontgomeryContext`,
//! `to_residue`/`mul_residue`, `mod_sqrt`, `PolyMod`.

use crate::shared::{calibrate, timed, Corpus, Digest};
use std::hint::black_box;

use rump::modular::{mod_inverse, mod_pow, mod_sqrt, BarrettContext, MontgomeryContext};
use rump::number_theory::{gcd, gcd_extended, jacobi};
use rump::polynomial::PolyZ;
use rump::{BigInt, BigUint};

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
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, m)| {
                    let mut p = m;
                    if !p.is_odd() {
                        p = p.add(&BigUint::one());
                    }
                    (a, p)
                })
                .collect();
            let results = data
                .iter()
                .map(|(a, p)| match mod_sqrt(a, p) {
                    Some(r) => r.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, p) in &data {
                        black_box(mod_sqrt(a, p));
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
        other => panic!("unknown case {other}"),
    }
}
