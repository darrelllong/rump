//! Corpus parsing, timing, calibration and digest machinery.
//!
//! This file is byte-identical between the two revision adapters: it touches no
//! Rump API. Only `cases.rs` differs, and only where the 0.3.0 rename forces
//! it.

use std::time::Instant;

/// A parsed corpus: groups of hex operands, in file order.
pub struct Corpus {
    pub header: Vec<(String, String)>,
    pub items: Vec<String>,
    /// Indices where a blank line ended a group, for the polynomial layout.
    pub groups: Vec<Vec<String>>,
}

pub fn load(path: &str) -> Corpus {
    let text = std::fs::read_to_string(path).expect("read corpus");
    let mut header = Vec::new();
    let mut items = Vec::new();
    let mut groups = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            if let Some((k, v)) = rest.split_once(": ") {
                header.push((k.to_string(), v.to_string()));
            }
            continue;
        }
        if line.trim().is_empty() {
            if !current.is_empty() {
                groups.push(std::mem::take(&mut current));
            }
            continue;
        }
        items.push(line.to_string());
        current.push(line.to_string());
    }
    if !current.is_empty() {
        groups.push(current);
    }
    Corpus {
        header,
        items,
        groups,
    }
}

impl Corpus {
    pub fn note(&self, key: &str) -> Option<&str> {
        self.header
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// A 64-bit FNV-1a digest over canonicalized result strings.
///
/// Deterministic and order-sensitive, which is what the comparison needs: two
/// revisions must produce the same results in the same order.
#[derive(Default)]
pub struct Digest(u64);

impl Digest {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    pub fn add(&mut self, s: &str) {
        for b in s.as_bytes() {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
        self.0 ^= 0xff;
        self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
    }
    pub fn finish(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Time `body` over `repeat` traversals, returning nanoseconds per operation.
///
/// `ops_per_traversal` converts to per-operation. The warmup is the same
/// declared shape in both adapters: one full traversal, discarded.
pub fn timed(repeat: usize, ops_per_traversal: usize, mut body: impl FnMut()) -> f64 {
    body(); // declared warmup, outside the measured interval
    let start = Instant::now();
    for _ in 0..repeat {
        body();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1e9 / (repeat as f64 * ops_per_traversal as f64)
}

/// Smallest repeat count whose measured interval reaches `target_ms`.
///
/// Calibration is outside the reported result: the driver calls this once and
/// then passes a fixed repeat to every measured child, so all readings for a
/// case do the same amount of work.
pub fn calibrate(target_ms: f64, mut body: impl FnMut()) -> usize {
    let mut repeat = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..repeat {
            body();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if ms >= target_ms || repeat > 1 << 30 {
            return repeat;
        }
        let factor = (target_ms / ms.max(1e-6)).ceil() as usize;
        repeat = (repeat * factor.max(2)).max(repeat + 1);
    }
}

/// SplitMix64 (Steele, Lea & Flood, OOPSLA 2014), the same generator the
/// corpus uses. Reseeded at the start of every timed traversal so both
/// revisions draw the identical byte stream and therefore take the identical
/// number of rejection rounds — otherwise the sampler comparison would be
/// measuring two different amounts of work.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn fill(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let word = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

/// The first `count` primes, by trial division. Small and deterministic, so
/// the smoothness base is identical in both adapters without a corpus file.
pub fn small_primes(count: usize) -> Vec<u64> {
    let mut primes: Vec<u64> = Vec::with_capacity(count);
    let mut candidate = 2u64;
    while primes.len() < count {
        if primes.iter().take_while(|p| *p * *p <= candidate).all(|p| candidate % p != 0) {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}
