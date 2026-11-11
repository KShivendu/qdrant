//! PQ codebook training benchmark: Lloyd k-means (current) vs e-means.
//!
//! Trains 256-centroid codebooks per sub-vector exactly as `EncodedVectorsPQ::find_centroids`
//! does (random sample, one k-means per chunk), encodes the whole base set, and measures
//! training time, reconstruction MSE and ADC recall@10 with and without exact rescoring.
//!
//! Input directory: `base.f32`, `query.f32` (row-major f32), `gt.u32` (nq x gt_k ids),
//! `meta.json` with `n`, `d`, `nq`, `metric` ("l2" or "dot"), `gt_k`.
//!
//! cargo run --release --example pq_codebook_bench -- <dir> <bucket> <threads> <seeds> <configs> [max_queries]
//! configs: comma list of `lloyd:<sample>:<iters>` or `emeans:<sample>:<epochs>[:norev|thr|qual]`.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use quantization::emeans::{BatchLaw, EmeansParams, emeans, resolve_recipe};
use quantization::kmeans::kmeans;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

const K: usize = 256;
const TOP: usize = 10;
const RESCORE: usize = 100;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs_err::read(path).unwrap();
    bytemuck_cast(&bytes)
}

fn bytemuck_cast<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let n = bytes.len() / size_of::<T>();
    let mut out = vec![T::default(); n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr().cast::<u8>(),
            n * size_of::<T>(),
        )
    };
    out
}

fn meta_field(meta: &str, key: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(meta).unwrap();
    v[key]
        .as_str()
        .map_or_else(|| v[key].to_string(), str::to_string)
}

#[derive(Clone, Copy)]
enum Algo {
    Lloyd {
        iters: usize,
    },
    Emeans {
        epochs: usize,
        revive: Option<f32>,
        law: BatchLaw,
    },
}

fn nearest(x: &[f32], centroids: &[f32], dim: usize) -> (u8, f32) {
    let mut best = (0u8, f32::MAX);
    for (i, c) in centroids.chunks_exact(dim).enumerate() {
        let d: f32 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.1 {
            best = (i as u8, d);
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let bucket: usize = args[2].parse().unwrap();
    let threads: usize = args[3].parse().unwrap();
    let seeds: u64 = args[4].parse().unwrap();
    let configs: Vec<(String, Algo, usize)> = args[5]
        .split(',')
        .map(|c| {
            let p: Vec<&str> = c.split(':').collect();
            let sample = p[1].parse().unwrap();
            let x = p[2].parse().unwrap();
            let algo = match p[0] {
                "lloyd" => Algo::Lloyd { iters: x },
                "emeans" => Algo::Emeans {
                    epochs: x,
                    // optional 4th field: norev | thr | qual
                    revive: (p.get(3) == Some(&"norev")).then_some(0.0),
                    law: match p.get(3) {
                        Some(&"thr") => BatchLaw::Throughput,
                        Some(&"qual") => BatchLaw::Quality,
                        _ => BatchLaw::Auto,
                    },
                },
                _ => panic!("unknown algo {}", p[0]),
            };
            (c.to_string(), algo, sample)
        })
        .collect();

    let meta = fs_err::read_to_string(format!("{dir}/meta.json")).unwrap();
    let d: usize = meta_field(&meta, "d").parse().unwrap();
    let nq_file: usize = meta_field(&meta, "nq").parse().unwrap();
    let nq = args
        .get(6)
        .map_or(nq_file, |a| a.parse::<usize>().unwrap().min(nq_file));
    let gt_k: usize = meta_field(&meta, "gt_k").parse().unwrap();
    let dot = meta_field(&meta, "metric") == "dot";
    let base = read_f32(&format!("{dir}/base.f32"));
    let mut query = read_f32(&format!("{dir}/query.f32"));
    query.truncate(nq * d);
    let gt: Vec<u32> = bytemuck_cast(&fs_err::read(format!("{dir}/gt.u32")).unwrap());
    let n = base.len() / d;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let division: Vec<std::ops::Range<usize>> = (0..d)
        .step_by(bucket)
        .map(|i| i..(i + bucket).min(d))
        .collect();
    let m = division.len();
    let stopped = AtomicBool::new(false);
    let ds = dir.trim_end_matches('/').rsplit('/').next().unwrap();
    eprintln!("{ds}: n={n} d={d} nq={nq} chunks={m} bucket={bucket} dot={dot}");

    for (name, algo, sample) in &configs {
        for seed in 0..seeds {
            let sample = (*sample).min(n);
            let mut rng = StdRng::seed_from_u64(seed);
            let mut rows = rand::seq::index::sample(&mut rng, n, sample).into_vec();
            rows.sort_unstable();

            // Train: one codebook per chunk, chunks sequential, like find_centroids.
            let t0 = Instant::now();
            let mut codebooks: Vec<Vec<f32>> = Vec::with_capacity(m);
            let mut recipe = None;
            for (ci, r) in division.iter().enumerate() {
                let mut sub = Vec::with_capacity(sample * r.len());
                for &i in &rows {
                    sub.extend_from_slice(&base[i * d + r.start..i * d + r.end]);
                }
                let cb = match *algo {
                    Algo::Lloyd { iters } => {
                        kmeans(&sub, K, r.len(), iters, threads, 1e-5, &stopped).unwrap()
                    }
                    Algo::Emeans {
                        epochs,
                        revive,
                        law,
                    } => {
                        let params = EmeansParams {
                            nepochs: epochs,
                            unused_threshold: revive,
                            batch_law: law,
                            seed: seed * 1000 + ci as u64,
                            ..Default::default()
                        };
                        recipe = Some(resolve_recipe(sample, K, r.len(), &params));
                        emeans(&sub, K, r.len(), &params, Some(&pool), &stopped).unwrap()
                    }
                };
                codebooks.push(cb);
            }
            let train_s = t0.elapsed().as_secs_f64();

            // Encode base.
            let t1 = Instant::now();
            let per_row: Vec<(Vec<u8>, f64)> = base
                .par_chunks_exact(d)
                .map(|x| {
                    let mut err = 0f64;
                    let codes = division
                        .iter()
                        .zip(&codebooks)
                        .map(|(r, cb)| {
                            let (c, e) = nearest(&x[r.clone()], cb, r.len());
                            err += f64::from(e);
                            c
                        })
                        .collect();
                    (codes, err)
                })
                .collect();
            let encode_s = t1.elapsed().as_secs_f64();
            let mse = per_row.iter().map(|p| p.1).sum::<f64>() / n as f64;
            let codes: Vec<u8> = per_row.into_iter().flat_map(|p| p.0).collect();
            let train_set: std::collections::HashSet<usize> = rows.iter().copied().collect();
            let mut held_err = 0f64;
            let mut held_n = 0usize;
            for i in (0..n).step_by((n / 20_000).max(1)) {
                if !train_set.contains(&i) {
                    let x = &base[i * d..(i + 1) * d];
                    for (r, cb) in division.iter().zip(&codebooks) {
                        held_err += f64::from(nearest(&x[r.clone()], cb, r.len()).1);
                    }
                    held_n += 1;
                }
            }
            let qerr: f64 = query
                .chunks_exact(d)
                .map(|q| {
                    division
                        .iter()
                        .zip(&codebooks)
                        .map(|(r, cb)| f64::from(nearest(&q[r.clone()], cb, r.len()).1))
                        .sum::<f64>()
                })
                .sum::<f64>()
                / nq as f64;

            // ADC search, recall@10 raw and after exact rescoring of top-100.
            // Also: PQ score error on each query's true top-10 vs on 10 random points.
            #[allow(clippy::type_complexity)]
            let (hits_raw, hits_rescored, nn_bias, nn_sq, rnd_bias, rnd_sq): (
                usize,
                usize,
                f64,
                f64,
                f64,
                f64,
            ) = query
                .par_chunks_exact(d)
                .enumerate()
                .map(|(qi, q)| {
                    let mut lut = vec![0f32; m * K];
                    for (ci, (r, cb)) in division.iter().zip(&codebooks).enumerate() {
                        let qs = &q[r.clone()];
                        for (j, c) in cb.chunks_exact(r.len()).enumerate() {
                            lut[ci * K + j] = if dot {
                                qs.iter().zip(c).map(|(a, b)| a * b).sum()
                            } else {
                                -qs.iter()
                                    .zip(c)
                                    .map(|(a, b)| (a - b) * (a - b))
                                    .sum::<f32>()
                            };
                        }
                    }
                    let mut scored: Vec<(f32, u32)> = codes
                        .chunks_exact(m)
                        .enumerate()
                        .map(|(i, cs)| {
                            let s: f32 = cs
                                .iter()
                                .enumerate()
                                .map(|(ci, &c)| lut[ci * K + c as usize])
                                .sum();
                            (s, i as u32)
                        })
                        .collect();
                    scored.select_nth_unstable_by(RESCORE, |a, b| b.0.total_cmp(&a.0));
                    scored.truncate(RESCORE);
                    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    let truth = &gt[qi * gt_k..qi * gt_k + TOP];
                    let raw = scored[..TOP]
                        .iter()
                        .filter(|s| truth.contains(&s.1))
                        .count();
                    let mut exact: Vec<(f32, u32)> = scored
                        .iter()
                        .map(|&(_, i)| {
                            let x = &base[i as usize * d..(i as usize + 1) * d];
                            let s = if dot {
                                q.iter().zip(x).map(|(a, b)| a * b).sum()
                            } else {
                                -q.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                            };
                            (s, i)
                        })
                        .collect();
                    exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    let rescored = exact[..TOP].iter().filter(|s| truth.contains(&s.1)).count();
                    let err = |i: usize| -> f64 {
                        let pq: f32 = codes[i * m..(i + 1) * m]
                            .iter()
                            .enumerate()
                            .map(|(ci, &c)| lut[ci * K + c as usize])
                            .sum();
                        let x = &base[i * d..(i + 1) * d];
                        let ex: f32 = if dot {
                            q.iter().zip(x).map(|(a, b)| a * b).sum()
                        } else {
                            -q.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                        };
                        f64::from(pq - ex)
                    };
                    let (mut nb, mut ns, mut rb, mut rs) = (0f64, 0f64, 0f64, 0f64);
                    for (k, &t) in truth.iter().enumerate() {
                        let e = err(t as usize);
                        nb += e;
                        ns += e * e;
                        let r = (qi * 7919 + k * 104_729) % n;
                        let e = err(r);
                        rb += e;
                        rs += e * e;
                    }
                    (raw, rescored, nb, ns, rb, rs)
                })
                .reduce(
                    || (0, 0, 0.0, 0.0, 0.0, 0.0),
                    |a, b| {
                        (
                            a.0 + b.0,
                            a.1 + b.1,
                            a.2 + b.2,
                            a.3 + b.3,
                            a.4 + b.4,
                            a.5 + b.5,
                        )
                    },
                );
            let denom = (nq * TOP) as f64;
            let (batch, alpha0) = recipe.map_or((0, 0.0), |r| (r.batch_size, r.alpha0));
            println!(
                "{{\"dataset\":\"{ds}\",\"bucket\":{bucket},\"config\":\"{name}\",\"seed\":{seed},\"sample\":{sample},\"threads\":{threads},\"train_s\":{train_s:.4},\"encode_s\":{encode_s:.3},\"mse_base\":{mse:.6},\"mse_heldout\":{:.6},\"mse_query\":{qerr:.6},\"recall10\":{:.4},\"recall10_rescore100\":{:.4},\"batch\":{batch},\"alpha0\":{alpha0:.4},\"nn_bias\":{:.6},\"nn_rmse\":{:.6},\"rnd_bias\":{:.6},\"rnd_rmse\":{:.6}}}",
                held_err / held_n.max(1) as f64,
                hits_raw as f64 / denom,
                hits_rescored as f64 / denom,
                nn_bias / denom,
                (nn_sq / denom).sqrt(),
                rnd_bias / denom,
                (rnd_sq / denom).sqrt(),
            );
        }
    }
}
