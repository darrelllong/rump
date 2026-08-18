//! Paired ABBA wrapper: one Pilot reading, one CSV row.
//!
//! Runs `baseline, candidate, candidate, baseline`, averages each pair, and
//! prints
//!
//! ```text
//! candidate_over_baseline,baseline_ns_per_op,candidate_ns_per_op,valid
//! ```
//!
//! The balanced order places the average observation time of both revisions at
//! the same point, which cancels first-order thermal and frequency drift: a
//! plain A,B ordering charges any monotone drift entirely to B.
//!
//! It exits non-zero rather than printing a timing when a child fails or the
//! four result digests disagree. Pilot must never receive a row it could treat
//! as a statistical sample when the two revisions did different work.

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
    let baseline = get("--baseline").expect("--baseline required");
    let candidate = get("--candidate").expect("--candidate required");
    let case = get("--case").expect("--case required");
    let corpus = get("--corpus").expect("--corpus required");
    let repeat: usize = get("--repeat")
        .expect("--repeat required")
        .parse()
        .expect("repeat is a number");

    // baseline, candidate, candidate, baseline.
    let sequence = [
        (&baseline, 'b'),
        (&candidate, 'c'),
        (&candidate, 'c'),
        (&baseline, 'b'),
    ];
    let mut base = Vec::new();
    let mut cand = Vec::new();
    let mut digests = Vec::new();
    for (program, which) in sequence {
        match run(program, &case, &corpus, repeat) {
            Ok(r) => {
                digests.push(r.digest.clone());
                if which == 'b' {
                    base.push(r.ns_per_op);
                } else {
                    cand.push(r.ns_per_op);
                }
            }
            Err(e) => {
                eprintln!("abba: {e}");
                std::process::exit(2);
            }
        }
    }

    // All four must agree: the two revisions computed the same results, in the
    // same order, from the same corpus.
    if digests.iter().any(|d| *d != digests[0]) {
        eprintln!("abba: result digests disagree: {digests:?}");
        std::process::exit(3);
    }

    let b = base.iter().sum::<f64>() / base.len() as f64;
    let c = cand.iter().sum::<f64>() / cand.len() as f64;
    if !(b.is_finite() && c.is_finite() && b > 0.0 && c > 0.0) {
        eprintln!("abba: non-positive mean (baseline {b}, candidate {c})");
        std::process::exit(4);
    }
    println!("{},{},{},1", c / b, b, c);
}
