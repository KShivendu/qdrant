# e-means for PQ codebook training

[e-means](https://e-means.github.io/) (Bessa et al. 2026, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501),
code in `faiss/contrib/e_means`) is mini-batch k-means with EMA centroid updates, a cosine step
schedule, batch size and step size from fitted scaling laws, dead-cluster revival by splitting, and
a final pass of exact means. `src/emeans.rs` is a Rust port of the reference NumPy version.

## Where k-means runs in Qdrant

| Place | Today | Does e-means help? |
|---|---|---|
| PQ codebooks (`encoded_vectors_pq.rs`) | Lloyd, 256 centroids per sub-vector, 10k sample, up to 100 iterations, chunks trained one after another | Yes, 2.3-2.6x faster PQ build on 1536-d, same recall (below) |
| TurboQuant (`turboquant/lloyd_max.rs`) | Lloyd-Max for N(0,1), constants solved once | No, the target distribution is known in closed form |
| Binary / scalar quantization | no clustering | No |

Qdrant has no IVF index. If one is ever added, its coarse quantizer (large k, millions of points)
is where the paper's 8-15x CPU speedups apply directly; PQ is the only k-means in the tree today.

## What changed

1. `emeans()` replaces Lloyd's `kmeans()` in `find_centroids`, 5 epochs, default recipe.
2. Codebooks for different chunks are trained in parallel, one chunk per thread, and samples for a
   group of chunks are gathered in one pass over the data (bounded at 64 MiB). Lloyd parallelised
   inside a chunk instead, which scales poorly: a chunk has 1-16 dims and 256 centroids.
3. Sample size stays at 10k (see "Bigger samples").

## Setup

- SIFT-1M (128-d, L2), GloVe-100 (1.18M, angular, normalised), dbpedia OpenAI ada-002 (first 3 of
  26 parquet shards, 100k base, 1000 held-out queries, 1536-d, normalised).
- Compression x16 (4 dims per chunk) and x64 (16 dims per chunk), 256 centroids per chunk.
- recall@10 of PQ ADC ranking against exact top-10, and after exact rescoring of the PQ top-100
  (what Qdrant does with `rescore` and oversampling 10).
- 4 cores, 15 GB RAM, no GPU. Codebook harness: 3 seeds each. End-to-end: 2 repeats, 300 queries.

Reproduce:

```sh
python3 examples/emeans/prep_datasets.py <data_dir>   # needs sift.hdf5, glove.hdf5, dbp_*.parquet
cargo run --release --example pq_codebook_bench -- <data_dir>/sift 4 4 3 lloyd:10000:100,emeans:10000:5
cargo run --release --example pq_e2e_bench -- <data_dir>/dbpedia 4 4 300 label
```

## End to end, through `EncodedVectorsPQ::encode`

Encode time is codebook training plus encoding every vector, 4 threads.

| dataset | x | variant | encode s | recall@10 | recall@10 rescored top-100 |
|---|---|---|---|---|---|
| dbpedia | x16 | current (Lloyd, 10k, 100 it) | 47.5 | 0.8548 | 1.0000 |
| dbpedia | x16 | e-means, 10k, 5 ep, parallel chunks (committed) | 18.6 | 0.8520 | 1.0000 |
| dbpedia | x16 | e-means, 100k, 5 ep, parallel chunks | 56.0 | 0.8565 | 1.0000 |
| dbpedia | x64 | current (Lloyd, 10k, 100 it) | 28.3 | 0.5615 | 0.9532 |
| dbpedia | x64 | e-means, 10k, 5 ep, parallel chunks (committed) | 12.1 | 0.5595 | 0.9522 |
| dbpedia | x64 | e-means, 100k, 5 ep, parallel chunks | 36.9 | 0.5474 | 0.9402 |
| glove | x16 | current (Lloyd, 10k, 100 it) | 17.1 | 0.5043 | 0.9344 |
| glove | x16 | e-means, 10k, 5 ep, parallel chunks (committed) | 15.2 | 0.5036 | 0.9318 |
| glove | x16 | e-means, 100k, 5 ep, parallel chunks | 18.5 | 0.5160 | 0.9350 |
| glove | x64 | current (Lloyd, 10k, 100 it) | 16.7 | 0.0883 | 0.2878 |
| glove | x64 | e-means, 10k, 5 ep, parallel chunks (committed) | 17.0 | 0.0933 | 0.2868 |
| glove | x64 | e-means, 100k, 5 ep, parallel chunks | 18.6 | 0.0937 | 0.2938 |
| sift | x16 | current (Lloyd, 10k, 100 it) | 17.7 | 0.6882 | 0.9979 |
| sift | x16 | e-means, 10k, 5 ep, parallel chunks (committed) | 13.5 | 0.6960 | 0.9975 |
| sift | x16 | e-means, 100k, 5 ep, parallel chunks | 17.5 | 0.7075 | 0.9982 |
| sift | x64 | current (Lloyd, 10k, 100 it) | 16.3 | 0.2790 | 0.7476 |
| sift | x64 | e-means, 10k, 5 ep, parallel chunks (committed) | 14.2 | 0.2985 | 0.7548 |
| sift | x64 | e-means, 100k, 5 ep, parallel chunks | 17.6 | 0.2973 | 0.7709 |

The committed variant (10k sample) cuts PQ build time by 2.6x (dbpedia x16, 47.5 s to 18.6 s) and 2.3x
(dbpedia x64), and by 1.1-1.3x on the 1M low-dim sets where encoding, not training, dominates.
Recall moves within run-to-run noise (about ±0.005 at 300 queries).

## Codebook harness

Same chunking and sampling as `find_centroids`, with chunks trained sequentially as the old code
did. Held-out MSE is on base rows not in the training sample, or on the queries when the sample
is the whole base (dbpedia at 100k). e-means here mostly ran on one thread, since its batches
(64-512 rows) fall below the parallel threshold, while Lloyd used 4.

| dataset | x | thr | config | train s | speedup | held-out MSE vs current | recall@10 | recall@10 rescored |
|---|---|---|---|---|---|---|---|---|
| sift | x16 | 4 | lloyd:10000:100 | 4.81 | 1.0x | +0.00% | 0.6949 ±0.0020 | 0.9974 |
| sift | x16 | 4 | lloyd:10000:25 | 1.86 | 2.6x | +0.62% | 0.6961 ±0.0003 | 0.9977 |
| sift | x16 | 4 | emeans:10000:5 | 0.88 | 5.4x | +0.23% | 0.6952 ±0.0011 | 0.9977 |
| sift | x16 | 4 | emeans:100000:5 | 5.69 | 0.8x | -11.23% | 0.7128 ±0.0031 | 0.9980 |
| sift | x16 | 4 | emeans:200000:5 | 10.28 | 0.5x | -13.23% | 0.7162 ±0.0017 | 0.9983 |
| sift | x16 | 4 | lloyd:100000:25 | 18.91 | 0.3x | -9.18% | 0.7105 ±0.0029 | 0.9982 |
| sift | x64 | 4 | lloyd:10000:100 | 1.84 | 1.0x | +0.00% | 0.2954 ±0.0030 | 0.7602 |
| sift | x64 | 4 | lloyd:10000:25 | 1.35 | 1.4x | +0.03% | 0.2948 ±0.0017 | 0.7586 |
| sift | x64 | 4 | emeans:10000:5 | 0.70 | 2.6x | -0.89% | 0.3008 ±0.0033 | 0.7599 |
| sift | x64 | 4 | emeans:100000:5 | 3.54 | 0.5x | -5.67% | 0.3126 ±0.0023 | 0.7817 |
| sift | x64 | 4 | emeans:200000:5 | 6.85 | 0.3x | -6.20% | 0.3198 ±0.0015 | 0.7837 |
| sift | x64 | 4 | lloyd:100000:25 | 14.78 | 0.1x | -5.00% | 0.3111 ±0.0037 | 0.7787 |
| glove | x16 | 4 | lloyd:10000:100 | 3.13 | 1.0x | +0.00% | 0.5105 ±0.0033 | 0.9343 |
| glove | x16 | 4 | lloyd:10000:25 | 1.61 | 1.9x | +0.07% | 0.5128 ±0.0031 | 0.9352 |
| glove | x16 | 4 | emeans:10000:5 | 0.73 | 4.3x | -0.20% | 0.5097 ±0.0057 | 0.9349 |
| glove | x16 | 4 | emeans:100000:5 | 4.73 | 0.7x | -3.92% | 0.5199 ±0.0045 | 0.9404 |
| glove | x16 | 4 | emeans:200000:5 | 7.92 | 0.4x | -4.53% | 0.5176 ±0.0013 | 0.9389 |
| glove | x16 | 4 | lloyd:100000:25 | 14.99 | 0.2x | -3.53% | 0.5239 ±0.0040 | 0.9378 |
| glove | x64 | 4 | lloyd:10000:100 | 1.61 | 1.0x | +0.00% | 0.0958 ±0.0005 | 0.2989 |
| glove | x64 | 4 | lloyd:10000:25 | 1.20 | 1.3x | +0.00% | 0.0962 ±0.0005 | 0.2992 |
| glove | x64 | 4 | emeans:10000:5 | 0.55 | 2.9x | -0.18% | 0.0958 ±0.0028 | 0.3006 |
| glove | x64 | 4 | emeans:100000:5 | 2.66 | 0.6x | -2.43% | 0.0964 ±0.0008 | 0.3097 |
| glove | x64 | 4 | emeans:200000:5 | 4.90 | 0.3x | -2.83% | 0.0980 ±0.0020 | 0.3071 |
| glove | x64 | 4 | lloyd:100000:25 | 9.18 | 0.2x | -2.18% | 0.0962 ±0.0022 | 0.3038 |
| dbpedia | x16 | 4 | lloyd:10000:100 | 39.22 | 1.0x | +0.00% | 0.8486 ±0.0008 | 0.9999 |
| dbpedia | x16 | 4 | lloyd:10000:25 | 21.45 | 1.8x | +0.05% | 0.8496 ±0.0013 | 0.9999 |
| dbpedia | x16 | 4 | emeans:10000:5 | 10.77 | 3.6x | -0.27% | 0.8510 ±0.0023 | 0.9999 |
| dbpedia | x16 | 4 | emeans:100000:5 | 62.55 | 0.6x | -3.69% | 0.8550 ±0.0007 | 0.9999 |
| dbpedia | x16 | 4 | emeans:200000:5 | 62.78 | 0.6x | -3.69% | 0.8550 ±0.0007 | 0.9999 |
| dbpedia | x16 | 4 | lloyd:100000:25 | 195.48 | 0.2x | -3.33% | 0.8577 ±0.0000 | 0.9999 |
| dbpedia | x64 | 4 | lloyd:10000:100 | 20.74 | 1.0x | +0.00% | 0.5614 ±0.0020 | 0.9532 |
| dbpedia | x64 | 4 | lloyd:10000:25 | 18.52 | 1.1x | +0.01% | 0.5621 ±0.0017 | 0.9526 |
| dbpedia | x64 | 4 | emeans:10000:5 | 7.59 | 2.7x | -0.30% | 0.5598 ±0.0034 | 0.9515 |
| dbpedia | x64 | 4 | emeans:100000:5 | 36.69 | 0.6x | -3.32% | 0.5494 ±0.0029 | 0.9452 |
| dbpedia | x64 | 4 | emeans:200000:5 | 39.51 | 0.5x | -3.32% | 0.5494 ±0.0029 | 0.9452 |
| dbpedia | x64 | 4 | lloyd:100000:25 | 151.81 | 0.1x | -3.01% | 0.5524 ±0.0000 | 0.9476 |
| sift | x16 | 1 | lloyd:10000:100 | 12.99 | 1.0x | +0.00% | 0.6923 ±0.0000 | 0.9971 |
| sift | x16 | 1 | emeans:10000:5 | 0.97 | 13.4x | +0.86% | 0.6961 ±0.0000 | 0.9982 |
| sift | x16 | 1 | emeans:200000:5 | 22.23 | 0.6x | -13.17% | 0.7157 ±0.0000 | 0.9987 |

Single-threaded, at the same 10k sample, e-means needs 13.4x less CPU than Lloyd on SIFT x16
(0.97 s vs 12.99 s).

## Bigger samples

The 10k sample is the bigger quality lever. With 256 centroids, it is 39 points per centroid. On SIFT
and GloVe, training on 100k-200k rows cuts held-out MSE by 3-13% and lifts rescored recall at x64
by 2.2 points on SIFT. e-means on 100k rows costs about as much as Lloyd on 10k, and Lloyd on 100k
costs 4-8x more for a worse result.

On dbpedia (100k x 1536-d) it goes the other way at x64. Query MSE drops 3.3%, but recall@10
drops 1.2 points raw and 0.8 points rescored, and Lloyd on 100k shows the same drop. So this
comes from the sample size, not from e-means. I have no explanation yet. Since lower MSE does not
mean better dot-product ranking here, the sample stays at 10k. Test sample size per collection size
and dimension before raising it.

## Not covered

- GPU (Qdrant's GPU path does not train PQ).
- Collections above 1.2M points, and other embedding models.
- Segment-level indexing time in a running Qdrant.
- The `stop_condition_u8` integration test fails with and without this change: in release mode,
  scalar encoding of 1M zero vectors finishes before the 100 ms stop signal fires.
