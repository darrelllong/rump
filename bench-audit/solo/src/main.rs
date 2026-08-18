//! Single-arm wrapper: one Pilot reading, one CSV row.
//!
//! Runs one adapter twice and prints
//!
//! ```text
//! ns_per_op,valid
//! ```
//!
//! This exists for the APIs v0.2.2 does not have, where there is no baseline to
//! pair against and the measurement is an absolute cost rather than a ratio.
//! There is no ABBA ordering to do here, and so no drift cancellation: an
//! absolute figure carries whatever drift the machine had, which is exactly why
//! these cells are reported as baselines and never as a comparison.
//!
//! Two runs rather than one, because a differing digest between them is the
//! only check available that the workload is deterministic; the paired wrapper
//! gets that check for free from the two revisions.
//!
//! Like the paired wrapper, it exits non-zero rather than printing a timing
//! Pilot could treat as a sample when the child failed or the digests disagree.

use std::process::Command;

struct Reading {
    ns_per_op: f64,
    digest: String,
}

fn run(program: &str, case: &str, corpus: &str, repeat: usize) -> Result<Reading, String> {
    let out = Command::new(program)
        .args([
            "--case", case, "--corpus", corpus, "--repeat", &repeat.to_string(), "--emit", "time",
        ])
        .output()
        .map_err(|e| format!("spawn {program}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{program} exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().last().ok_or("no output")?;
    let mut parts = line.split_whitespace();
    let ns: f64 = parts
        .next()
        .ok_or("no timing field")?
        .parse()
        .map_err(|e| format!("timing not a number: {e}"))?;
    let digest = parts.next().ok_or("no digest field")?.to_string();
    if !ns.is_finite() || ns <= 0.0 {
        return Err(format!("non-positive or non-finite timing {ns}"));
    }
    Ok(Reading {
        ns_per_op: ns,
        digest,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let program = get("--program").expect("--program required");
    let case = get("--case").expect("--case required");
    let corpus = get("--corpus").expect("--corpus required");
    let repeat: usize = get("--repeat")
        .expect("--repeat required")
        .parse()
        .expect("repeat is a number");

    let mut times = Vec::new();
    let mut digests = Vec::new();
    for _ in 0..2 {
        match run(&program, &case, &corpus, repeat) {
            Ok(r) => {
                digests.push(r.digest);
                times.push(r.ns_per_op);
            }
            Err(e) => {
                eprintln!("solo: {e}");
                std::process::exit(2);
            }
        }
    }
    if digests[0] != digests[1] {
        eprintln!("solo: result digests disagree: {digests:?}");
        std::process::exit(3);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    if !(mean.is_finite() && mean > 0.0) {
        eprintln!("solo: non-positive mean {mean}");
        std::process::exit(4);
    }
    println!("{mean},1");
}
