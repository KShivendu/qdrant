//! End-to-end PQ benchmark through `EncodedVectorsPQ::encode`: wall time of codebook training
//! plus encoding, and ADC recall@10 (raw and after exact rescoring of the top 100).
//!
//! cargo run --release --example pq_e2e_bench -- <dir> <bucket> <threads> <nq> <label>

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use quantization::encoded_storage::TestEncodedStorageBuilder;
use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
use quantization::encoded_vectors_pq::{self, EncodedVectorsPQ};
use rayon::prelude::*;

fn read<T: Copy + Default>(path: &str) -> Vec<T> {
    let bytes = fs_err::read(path).unwrap();
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, bucket, threads, nq, label) = (
        &a[1],
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<usize>().unwrap(),
        a[4].parse::<usize>().unwrap(),
        &a[5],
    );
    let meta: serde_json::Value =
        serde_json::from_str(&fs_err::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let d = meta["d"].as_u64().unwrap() as usize;
    let gt_k = meta["gt_k"].as_u64().unwrap() as usize;
    let dot = meta["metric"] == "dot";
    let base: Vec<f32> = read(&format!("{dir}/base.f32"));
    let query: Vec<f32> = read(&format!("{dir}/query.f32"));
    let gt: Vec<u32> = read(&format!("{dir}/gt.u32"));
    let n = base.len() / d;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let params = VectorParameters {
        dim: d,
        deprecated_count: None,
        distance_type: if dot {
            DistanceType::Dot
        } else {
            DistanceType::L2
        },
        invert: !dot,
    };
    let size = encoded_vectors_pq::get_quantized_vector_size(&params, bucket);
    let t0 = Instant::now();
    let encoded = EncodedVectorsPQ::encode(
        base.chunks_exact(d),
        TestEncodedStorageBuilder::new(None, size),
        &params,
        n,
        bucket,
        threads,
        None,
        &AtomicBool::new(false),
    )
    .unwrap();
    let encode_s = t0.elapsed().as_secs_f64();

    let (raw, rescored): (usize, usize) = query
        .par_chunks_exact(d)
        .take(nq)
        .enumerate()
        .map(|(qi, q)| {
            let counter = HardwareCounterCell::new();
            let eq = encoded.encode_query(q);
            let mut s: Vec<(f32, u32)> = (0..n as u32)
                .map(|i| (encoded.score_point(&eq, i, &counter), i))
                .collect();
            s.select_nth_unstable_by(100, |a, b| b.0.total_cmp(&a.0));
            s.truncate(100);
            s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let truth = &gt[qi * gt_k..qi * gt_k + 10];
            let raw = s[..10].iter().filter(|x| truth.contains(&x.1)).count();
            let mut e: Vec<(f32, u32)> = s
                .iter()
                .map(|&(_, i)| {
                    let x = &base[i as usize * d..(i as usize + 1) * d];
                    let v = if dot {
                        q.iter().zip(x).map(|(a, b)| a * b).sum()
                    } else {
                        -q.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                    };
                    (v, i)
                })
                .collect();
            e.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            (raw, e[..10].iter().filter(|x| truth.contains(&x.1)).count())
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let ds = dir.trim_end_matches('/').rsplit('/').next().unwrap();
    let den = (nq * 10) as f64;
    println!(
        "{{\"label\":\"{label}\",\"dataset\":\"{ds}\",\"bucket\":{bucket},\"threads\":{threads},\"encode_s\":{encode_s:.3},\"recall10\":{:.4},\"recall10_rescore100\":{:.4}}}",
        raw as f64 / den,
        rescored as f64 / den
    );
}
