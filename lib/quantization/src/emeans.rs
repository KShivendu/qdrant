//! e-means: mini-batch k-means with EMA centroid updates.
//!
//! Port of Algorithm 1 from "e-means: outperforming k-means (in one-tenth the epochs)",
//! Bessa et al. 2026, <https://arxiv.org/abs/2611.09501>, following the reference
//! implementation in `faiss/contrib/e_means` (`base.py`, `e_means.py`).
//!
//! Each cluster keeps an EMA mass `N_j` and position `c_j = S_j / N_j`. For each mini-batch
//! with per-cluster counts `n_j` and sums `s_j`:
//!
//! ```text
//! N_j <- (1 - a) N_j + a n_j
//! c_j <- c_j + (a n_j / N_j) (s_j / n_j - c_j)
//! ```
//!
//! `a` follows a cosine schedule from `alpha0` to `0.1 alpha0`. Batch size and `alpha0` come
//! from the paper's fitted scaling laws. Every 5 updates, clusters whose mass is below 10% of
//! the mean are revived by splitting a donor sampled in proportion to its mass. The returned
//! centroids are the plain means of the assignments made during the last epoch.

use std::sync::atomic::{AtomicBool, Ordering};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::EncodingError;

const ALPHA_MIN_FACTOR: f64 = 0.1;
const CHECK_UNUSED_EVERY: usize = 5;
const INIT_USAGE: f32 = 1.0 / 65536.0;
const SPLIT_EPSILON: f32 = 1e-4;
const UNUSED_THRESHOLD: f32 = 0.1;
const NO_REVIVAL_MAX_D: usize = 2;
const AUTO_QUALITY_MAX_D: usize = 16;
/// (C_B, b_k, b_n, b_T) for `B = C_B k^b_k n^b_n T^b_T`.
const BATCH_LAW_THROUGHPUT: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const BATCH_LAW_QUALITY: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// (C_a, a_k, a_B, a_n, a_T) for `alpha0 = C_a k^a_k B^a_B n^a_n T^a_T`.
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);
/// Below this many multiply-adds a batch is assigned on the calling thread.
const PARALLEL_MIN_WORK: usize = 1 << 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLaw {
    /// `Quality` for `dim <= 16`, `Throughput` above.
    Auto,
    Throughput,
    /// Batches ~8x smaller than `Throughput`.
    Quality,
}

#[derive(Debug, Clone)]
pub struct EmeansParams {
    pub nepochs: usize,
    pub batch_law: BatchLaw,
    /// Overrides the batch law.
    pub batch_size: Option<usize>,
    /// Overrides the step law.
    pub alpha0: Option<f64>,
    pub seed: u64,
}

impl Default for EmeansParams {
    fn default() -> Self {
        Self {
            nepochs: 5,
            batch_law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            seed: 1234,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmeansRecipe {
    pub batch_size: usize,
    pub alpha0: f64,
    pub total_steps: usize,
    pub unused_threshold: f32,
}

fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    (if value / low < high / value {
        low
    } else {
        high
    }) as usize
}

pub fn resolve_recipe(n: usize, k: usize, dim: usize, params: &EmeansParams) -> EmeansRecipe {
    let (nf, kf, tf) = (n as f64, k as f64, params.nepochs as f64);
    let law = match params.batch_law {
        BatchLaw::Auto if dim <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto | BatchLaw::Throughput => BatchLaw::Throughput,
        BatchLaw::Quality => BatchLaw::Quality,
    };
    let batch_size = match params.batch_size {
        Some(b) => b.min(n),
        None => {
            let (c, bk, bn, bt) = match law {
                BatchLaw::Quality => BATCH_LAW_QUALITY,
                BatchLaw::Auto | BatchLaw::Throughput => BATCH_LAW_THROUGHPUT,
            };
            let b = round_power_of_two(c * kf.powf(bk) * nf.powf(bn) * tf.powf(bt));
            let cap = match law {
                BatchLaw::Quality => n / 2,
                BatchLaw::Auto | BatchLaw::Throughput => round_power_of_two(nf / 2.0),
            };
            b.min(cap).min(n).max(1)
        }
    };
    let alpha0 = params.alpha0.unwrap_or_else(|| {
        let (c, ak, ab, an, at) = ALPHA_LAW;
        (c * kf.powf(ak) * (batch_size as f64).powf(ab) * nf.powf(an) * tf.powf(at))
            .clamp(1e-4, 1.0)
    });
    let steps_per_epoch = n.div_ceil(batch_size);
    EmeansRecipe {
        batch_size,
        alpha0,
        total_steps: steps_per_epoch * params.nepochs,
        unused_threshold: if dim <= NO_REVIVAL_MAX_D {
            0.0
        } else {
            UNUSED_THRESHOLD
        },
    }
}

fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

#[inline]
fn nearest(vector: &[f32], centroids: &[f32], dim: usize) -> u32 {
    let mut best = f32::MAX;
    let mut best_idx = 0;
    for (idx, c) in centroids.chunks_exact(dim).enumerate() {
        let d: f32 = vector.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best {
            best = d;
            best_idx = idx;
        }
    }
    best_idx as u32
}

/// Train `centroids_count` centroids on `data` (row-major, `dim` columns) with e-means.
///
/// Returns flattened centroids, `centroids_count * dim` values.
pub fn emeans(
    data: &[f32],
    centroids_count: usize,
    dim: usize,
    params: &EmeansParams,
    pool: Option<&rayon::ThreadPool>,
    stopped: &AtomicBool,
) -> Result<Vec<f32>, EncodingError> {
    let k = centroids_count;
    let n = data.len() / dim;
    if n < k {
        return Err(EncodingError::ArgumentsError(format!(
            "e-means needs at least {k} vectors, got {n}"
        )));
    }
    let recipe = resolve_recipe(n, k, dim, params);
    let mut rng = StdRng::seed_from_u64(params.seed);

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(&mut rng);
    let mut centroids = Vec::with_capacity(k * dim);
    for &i in &order[..k] {
        centroids.extend_from_slice(&data[i as usize * dim..(i as usize + 1) * dim]);
    }
    let mut usage = vec![INIT_USAGE; k];

    let mut batch_sums = vec![0f32; k * dim];
    let mut batch_counts = vec![0u32; k];
    let mut pass_sums = vec![0f64; k * dim];
    let mut pass_counts = vec![0u64; k];
    let mut labels = vec![0u32; recipe.batch_size];

    let mut step = 0;
    for epoch in 0..params.nepochs {
        let last_epoch = epoch + 1 == params.nepochs;
        order.shuffle(&mut rng);
        for batch in order.chunks(recipe.batch_size) {
            if stopped.load(Ordering::Relaxed) {
                return Err(EncodingError::Stopped);
            }
            let labels = &mut labels[..batch.len()];
            let assign = |(label, &row): (&mut u32, &u32)| {
                let row = row as usize;
                *label = nearest(&data[row * dim..(row + 1) * dim], &centroids, dim);
            };
            match pool {
                Some(pool) if batch.len() * k * dim >= PARALLEL_MIN_WORK => pool.install(|| {
                    labels
                        .par_iter_mut()
                        .zip(batch.par_iter())
                        .with_min_len(64)
                        .for_each(assign)
                }),
                _ => labels.iter_mut().zip(batch.iter()).for_each(assign),
            }

            batch_sums.fill(0.0);
            batch_counts.fill(0);
            for (&label, &row) in labels.iter().zip(batch) {
                let (label, row) = (label as usize, row as usize);
                batch_counts[label] += 1;
                let vector = &data[row * dim..(row + 1) * dim];
                for (s, v) in batch_sums[label * dim..(label + 1) * dim]
                    .iter_mut()
                    .zip(vector)
                {
                    *s += v;
                }
                if last_epoch {
                    pass_counts[label] += 1;
                    for (s, v) in pass_sums[label * dim..(label + 1) * dim]
                        .iter_mut()
                        .zip(vector)
                    {
                        *s += f64::from(*v);
                    }
                }
            }

            let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps) as f32;
            let replace = alpha >= 1.0;
            if !replace {
                usage.iter_mut().for_each(|u| *u *= 1.0 - alpha);
            }
            for j in 0..k {
                let count = batch_counts[j];
                if count == 0 {
                    continue;
                }
                let count = count as f32;
                let mass = if replace { 0.0 } else { usage[j] } + alpha * count;
                let weight = alpha * count / mass;
                let c = &mut centroids[j * dim..(j + 1) * dim];
                let s = &batch_sums[j * dim..(j + 1) * dim];
                if weight >= 1.0 {
                    c.iter_mut().zip(s).for_each(|(c, s)| *c = s / count);
                } else {
                    c.iter_mut()
                        .zip(s)
                        .for_each(|(c, s)| *c += weight * (s / count - *c));
                }
                usage[j] = mass;
            }

            step += 1;
            if recipe.unused_threshold > 0.0 && step % CHECK_UNUSED_EVERY == 0 {
                revive_unused(
                    &mut centroids,
                    &mut usage,
                    dim,
                    recipe.unused_threshold,
                    &mut rng,
                );
            }
        }
    }

    for j in 0..k {
        if pass_counts[j] > 0 {
            let count = pass_counts[j] as f64;
            for (c, s) in centroids[j * dim..(j + 1) * dim]
                .iter_mut()
                .zip(&pass_sums[j * dim..(j + 1) * dim])
            {
                *c = (s / count) as f32;
            }
        }
    }
    Ok(centroids)
}

/// Split donors, sampled without replacement in proportion to their mass, into clusters whose
/// mass fell below `threshold` times the mean mass. Returns the number of splits.
fn revive_unused(
    centroids: &mut [f32],
    usage: &mut [f32],
    dim: usize,
    threshold: f32,
    rng: &mut StdRng,
) -> usize {
    let k = usage.len();
    let limit = usage.iter().sum::<f32>() * threshold / k as f32;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < limit).collect();
    if unused.is_empty() {
        return 0;
    }
    // Gumbel top-k: keys ln(N_j) - ln(-ln(u)) select donors proportionally to N_j.
    let mut donors: Vec<(f64, usize)> = (0..k)
        .filter(|&j| usage[j] >= limit && usage[j] > 0.0)
        .map(|j| {
            let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            (f64::from(usage[j]).ln() - (-u.ln()).ln(), j)
        })
        .collect();
    let nsplit = unused.len().min(donors.len());
    if nsplit == 0 {
        return 0;
    }
    donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    for (&(_, source), &target) in donors.iter().zip(&unused).take(nsplit) {
        let half = usage[source] / 2.0;
        usage[source] = half;
        usage[target] = half;
        for i in 0..dim {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let value = centroids[source * dim + i];
            centroids[source * dim + i] = value * (1.0 - SPLIT_EPSILON * sign);
            centroids[target * dim + i] = value * (1.0 + SPLIT_EPSILON * sign);
        }
        if centroids[source * dim..(source + 1) * dim]
            == centroids[target * dim..(target + 1) * dim]
        {
            let v = centroids[target * dim];
            centroids[target * dim] = if v.next_up().is_finite() {
                v.next_up()
            } else {
                v.next_down()
            };
        }
    }
    nsplit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_matches_reference() {
        // Values from faiss `_resolve_recipe` / `_step_law` for n=1M, k=16384, T=5, d=96
        // (throughput law): raw batch 0.035134 * 16384^0.1347 * 1e6^0.8765 * 5^0.8133.
        let r = resolve_recipe(1_000_000, 16384, 96, &EmeansParams::default());
        let raw = 0.035134 * 16384f64.powf(0.134749) * 1e6f64.powf(0.876473) * 5f64.powf(0.813344);
        assert_eq!(r.batch_size, round_power_of_two(raw));
        assert!(r.alpha0 > 0.0 && r.alpha0 <= 1.0);
        assert_eq!(r.unused_threshold, UNUSED_THRESHOLD);
        assert_eq!(
            resolve_recipe(10_000, 256, 2, &EmeansParams::default()).unused_threshold,
            0.0
        );
    }

    fn mse(data: &[f32], centroids: &[f32], dim: usize) -> f64 {
        let total: f64 = data
            .chunks_exact(dim)
            .map(|x| {
                let c = nearest(x, centroids, dim) as usize;
                let c = &centroids[c * dim..(c + 1) * dim];
                f64::from(x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f32>())
            })
            .sum();
        total / (data.len() / dim) as f64
    }

    /// Same setting as Qdrant PQ: 256 centroids on a 10k sample of 8-dim sub-vectors.
    /// Five e-means epochs must land within 1% of 100 Lloyd iterations on held-out data.
    #[test]
    fn matches_lloyd_on_held_out() {
        let dim = 8;
        let mut rng = StdRng::seed_from_u64(7);
        let mut gaussian = || {
            let (u1, u2): (f32, f32) = (rng.random::<f32>().max(1e-12), rng.random());
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };
        // Anisotropic Gaussian: per-coordinate scale 1..dim.
        let mut sample = |n: usize| -> Vec<f32> {
            (0..n * dim)
                .map(|i| gaussian() * (1 + i % dim) as f32)
                .collect()
        };
        let train = sample(10_000);
        let test = sample(10_000);
        let stopped = AtomicBool::new(false);
        let e = emeans(&train, 256, dim, &EmeansParams::default(), None, &stopped).unwrap();
        let l = crate::kmeans::kmeans(&train, 256, dim, 100, 1, 1e-5, &stopped).unwrap();
        let (e, l) = (mse(&test, &e, dim), mse(&test, &l, dim));
        assert!(e < l * 1.01, "e-means {e} vs Lloyd {l}");
    }

    #[test]
    fn stops() {
        let data = vec![0.5f32; 4 * 1000];
        let stopped = AtomicBool::new(true);
        let r = emeans(&data, 16, 4, &EmeansParams::default(), None, &stopped);
        assert_eq!(r, Err(EncodingError::Stopped));
    }
}
