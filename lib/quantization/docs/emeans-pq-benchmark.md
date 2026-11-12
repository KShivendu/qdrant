# e-means for PQ codebook training

[e-means](https://e-means.github.io/) (Bessa et al. 2026, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501),
code in `faiss/contrib/e_means`) is mini-batch k-means with EMA centroid updates, a cosine step
schedule, batch and step size from fitted scaling laws, dead-cluster revival by splitting, and a
final pass of exact means. `src/emeans.rs` ports the reference NumPy version.

## Verdict

- PQ codebook training is the only k-means in Qdrant. TurboQuant's Lloyd-Max levels are closed-form
  constants for N(0,1), and BQ/SQ do no clustering.
- **x4/x8/x16 (chunks of 1-4 dims): e-means ships.** PQ build on dbpedia-openai-100K x16 drops from
  57.8 s to 20.0 s (2.9x). Server precision@10 matches at all 12 search settings (diff at most 0.001).
- **x32/x64: Lloyd stays.** With 16-dim chunks, e-means cost 0.3-0.5 points of rescored recall@10 on
  OpenAI embeddings in every offline test (100K: 0.9309 vs 0.9345, 5 seeds; 1M: 0.9180 vs 0.9215),
  and about 1 point in the server test, where the CIs overlap. Turning revival off, switching the
  batch law, 20 epochs and 5 polishing Lloyd iterations did not close the gap; 20 polishing
  iterations closed half of it. x32 was not tested, so it stays on Lloyd too.
  (`EMEANS_MAX_CHUNK_DIM = 4`.)
- **Bigger training samples hurt OpenAI x64 recall** at 100K and at 1M, with both algorithms, even
  though MSE drops 2-3%. On SIFT and GloVe they help (see the appendix). The sample stays at 10k.
- **The main PQ error on OpenAI embeddings is bias, not k-means quality.** At x64, PQ underestimates
  the score of each query's true top-10 by 0.065 (100K) and 0.082 (1M), against 0.000 for random
  points. Every codebook variant tested has the same bias. Score-aware (anisotropic,
  ScaNN-style) quantization targets exactly this. It is untested here and is the bigger lever.

## Setup

- [vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark) datasets `dbpedia-openai-100K-1536-angular`
  (100,000 base, 5,000 queries) and `dbpedia-openai-1M-1536-angular` (975,000 base, 5,000 queries),
  with the GT shipped in `tests.jsonl`, spot-checked against exact search.
- 4 vCPU, 15 GB RAM. Interleaved old/new runs; 95% CIs (t-distribution) over runs.
- "old" = parent commit (Lloyd, 100 iterations, chunks sequential, 4 threads inside each chunk).
  "new10k"/"par-emeans" = e-means, 10k sample, chunks in parallel. "final" = committed code.

## Server test: vector-db-benchmark against a running Qdrant

`qdrant` built with `--profile perf` from both commits. The vector-db-benchmark config (not in
this repo) mirrors upstream `qdrant-quantizations.json` (m=16, ef_construct=128, 2 segments, always_ram) with
`product` x16 and x64. 12 search settings, 5000 queries each. 3 runs per binary, order old, new, new,
old, old, new. Indexing time is from upload end to the collection turning green with every vector
indexed, polled every 0.2 s.

### Indexing time after upload (s), precise poll
x16: old 325.9 ±125.1 (n=3), new 257.5 ±79.5 (n=3), diff -68.4 ±109.6  runs old=[355, 268, 355] new=[291, 255, 227]
x64: old 156.9 ±52.1 (n=3), new 122.2 ±4.6 (n=3), diff -34.6 ±52.3  runs old=[145, 181, 144] new=[120, 123, 124]
### vector-db-benchmark total import time (s, 5 s poll granularity)
x16: old 347.6 ±122.9, new 281.5 ±81.0
x64: old 180.5 ±54.4, new 145.2 ±1.3
### precision@10 (5000 queries), mean ±95% CI over runs; diff = new - old
| x | hnsw_ef | rescore | oversampling | old | new | diff |
|---|---|---|---|---|---|---|
| x16 | 64 | False | - | 0.8087 ±0.0062 | 0.8080 ±0.0023 | -0.0006 ±0.0049 |
| x16 | 64 | True | 1.0 | 0.8087 ±0.0062 | 0.8080 ±0.0023 | -0.0006 ±0.0049 |
| x16 | 64 | True | 2.0 | 0.9472 ±0.0059 | 0.9473 ±0.0033 | +0.0001 ±0.0050 |
| x16 | 64 | True | 4.0 | 0.9620 ±0.0054 | 0.9625 ±0.0037 | +0.0005 ±0.0042 |
| x16 | 128 | False | - | 0.8189 ±0.0049 | 0.8182 ±0.0033 | -0.0008 ±0.0038 |
| x16 | 128 | True | 1.0 | 0.8189 ±0.0049 | 0.8182 ±0.0033 | -0.0008 ±0.0038 |
| x16 | 128 | True | 2.0 | 0.9653 ±0.0030 | 0.9655 ±0.0032 | +0.0001 ±0.0028 |
| x16 | 128 | True | 4.0 | 0.9828 ±0.0023 | 0.9827 ±0.0030 | -0.0001 ±0.0025 |
| x16 | 256 | False | - | 0.8241 ±0.0032 | 0.8231 ±0.0029 | -0.0010 ±0.0028 |
| x16 | 256 | True | 1.0 | 0.8241 ±0.0032 | 0.8231 ±0.0029 | -0.0010 ±0.0028 |
| x16 | 256 | True | 2.0 | 0.9739 ±0.0013 | 0.9739 ±0.0018 | +0.0000 ±0.0014 |
| x16 | 256 | True | 4.0 | 0.9923 ±0.0002 | 0.9923 ±0.0013 | +0.0000 ±0.0013 |
| x64 | 64 | False | - | 0.4872 ±0.0058 | 0.4798 ±0.0119 | -0.0074 ±0.0098 |
| x64 | 64 | True | 1.0 | 0.4872 ±0.0058 | 0.4798 ±0.0119 | -0.0074 ±0.0098 |
| x64 | 64 | True | 2.0 | 0.6500 ±0.0045 | 0.6379 ±0.0182 | -0.0121 ±0.0187 |
| x64 | 64 | True | 4.0 | 0.7703 ±0.0058 | 0.7584 ±0.0186 | -0.0119 ±0.0195 |
| x64 | 128 | False | - | 0.4978 ±0.0056 | 0.4914 ±0.0131 | -0.0064 ±0.0106 |
| x64 | 128 | True | 1.0 | 0.4978 ±0.0056 | 0.4914 ±0.0131 | -0.0064 ±0.0106 |
| x64 | 128 | True | 2.0 | 0.6670 ±0.0050 | 0.6565 ±0.0193 | -0.0105 ±0.0200 |
| x64 | 128 | True | 4.0 | 0.7950 ±0.0047 | 0.7854 ±0.0186 | -0.0096 ±0.0192 |
| x64 | 256 | False | - | 0.5038 ±0.0053 | 0.4976 ±0.0142 | -0.0062 ±0.0112 |
| x64 | 256 | True | 1.0 | 0.5038 ±0.0053 | 0.4976 ±0.0142 | -0.0062 ±0.0112 |
| x64 | 256 | True | 2.0 | 0.6765 ±0.0062 | 0.6663 ±0.0205 | -0.0102 ±0.0214 |
| x64 | 256 | True | 4.0 | 0.8093 ±0.0036 | 0.7998 ±0.0208 | -0.0095 ±0.0211 |

Indexing is faster with e-means in every pairing except one old run (268 s), but at n=3 the
x16 interval includes zero. HNSW construction dominates indexing time.

## End to end, through `EncodedVectorsPQ::encode` (dbpedia-openai-100K, 2000 queries)

Encode time = codebook training + encoding all vectors. `par-*` variants select the trainer in an
experiment build (not committed): `lloyd100`/`lloyd25` = Lloyd per chunk with chunks in parallel;
`emeans_polishN` = e-means then N Lloyd iterations.

| dataset | x | variant | n | encode s | recall@10 | rescored top-100 |
|---|---|---|---|---|---|---|
| vdb100k | x16 | final | 2 | 20.0 ±1.8 | 0.8226 ±0.0044 | 1.0000 ±0.0000 |
| vdb100k | x16 | new | 3 | 62.0 ±3.8 | 0.8292 ±0.0030 | 1.0000 ±0.0000 |
| vdb100k | x16 | new10k | 3 | 20.0 ±0.8 | 0.8259 ±0.0048 | 1.0000 ±0.0000 |
| vdb100k | x16 | old | 3 | 57.8 ±2.4 | 0.8238 ±0.0019 | 1.0000 ±0.0000 |
| vdb100k | x16 | par-emeans | 3 | 19.7 ±1.4 | 0.8258 ±0.0024 | 1.0000 ±0.0001 |
| vdb100k | x16 | par-emeans_polish20 | 3 | 34.8 ±0.7 | 0.8254 ±0.0047 | 1.0000 ±0.0000 |
| vdb100k | x16 | par-emeans_polish5 | 3 | 24.4 ±0.5 | 0.8238 ±0.0074 | 1.0000 ±0.0000 |
| vdb100k | x16 | par-lloyd100 | 3 | 75.0 ±11.8 | 0.8248 ±0.0061 | 1.0000 ±0.0000 |
| vdb100k | x16 | par-lloyd25 | 3 | 42.1 ±1.9 | 0.8262 ±0.0061 | 1.0000 ±0.0000 |
| vdb100k | x64 | final | 2 | 32.2 ±10.3 | 0.5090 ±0.0114 | 0.9345 ±0.0210 |
| vdb100k | x64 | new | 3 | 35.8 ±1.1 | 0.5019 ±0.0092 | 0.9235 ±0.0021 |
| vdb100k | x64 | new10k | 3 | 12.4 ±2.1 | 0.5055 ±0.0080 | 0.9314 ±0.0072 |
| vdb100k | x64 | old | 3 | 32.0 ±3.9 | 0.5101 ±0.0060 | 0.9363 ±0.0033 |
| vdb100k | x64 | par-emeans | 3 | 12.3 ±1.3 | 0.5038 ±0.0086 | 0.9309 ±0.0030 |
| vdb100k | x64 | par-emeans_polish20 | 3 | 19.3 ±1.2 | 0.5060 ±0.0005 | 0.9326 ±0.0023 |
| vdb100k | x64 | par-emeans_polish5 | 3 | 15.5 ±0.8 | 0.5055 ±0.0079 | 0.9303 ±0.0050 |
| vdb100k | x64 | par-lloyd100 | 3 | 39.9 ±1.7 | 0.5071 ±0.0093 | 0.9332 ±0.0031 |
| vdb100k | x64 | par-lloyd25 | 3 | 28.5 ±1.1 | 0.5117 ±0.0048 | 0.9370 ±0.0060 |

Training chunks in parallel only helps e-means. Lloyd with chunks in parallel is slower than the
old layout, so the x16 speedup comes from e-means.

## dbpedia-openai-1M end to end (500 queries, 2 runs)

| dataset | x | variant | n | encode s | recall@10 | rescored top-100 |
|---|---|---|---|---|---|---|
| vdb1m | x16 | new10k | 2 | 154.7 ±22.7 | 0.8165 ±0.0623 | 0.9998 ±0.0000 |
| vdb1m | x16 | old | 2 | 180.8 ±2.4 | 0.8160 ±0.0381 | 0.9998 ±0.0025 |
| vdb1m | x64 | new10k | 2 | 94.6 ±33.2 | 0.5136 ±0.0585 | 0.9121 ±0.0013 |
| vdb1m | x64 | old | 2 | 117.7 ±1.8 | 0.5068 ±0.0585 | 0.9173 ±0.0648 |

At 1M, encoding the vectors dominates, so the training saving is 20-25 s of 118-180 s.

## Codebook harness (dbpedia-openai-100K, 5 seeds, 2000 queries)

Chunks trained sequentially, as in the old code. MSE is measured on the 5000 held-out queries.

| dataset | x | config | n | train s | held-out MSE vs current (queries for vdb*) | recall@10 | recall@10 rescored top-100 |
|---|---|---|---|---|---|---|---|
| vdb100k | x16 | lloyd:10000:100 | 5 | 45.5 | -0.00% ±0.06 | 0.8229 ±0.0021 | 1.0000 ±0.0000 |
| vdb100k | x16 | lloyd:10000:25 | 5 | 21.6 | +0.06% ±0.06 | 0.8233 ±0.0020 | 1.0000 ±0.0000 |
| vdb100k | x16 | emeans:10000:5 | 5 | 9.4 | +0.36% ±0.09 | 0.8238 ±0.0028 | 1.0000 ±0.0000 |
| vdb100k | x16 | emeans:50000:5 | 5 | 43.0 | -2.61% ±0.10 | 0.8280 ±0.0013 | 1.0000 ±0.0000 |
| vdb100k | x16 | emeans:100000:5 | 5 | 67.2 | -3.40% ±0.03 | 0.8304 ±0.0013 | 1.0000 ±0.0000 |
| vdb100k | x16 | lloyd:100000:25 | 5 | 189.3 | -3.38% ±0.00 | 0.8279 ±0.0003 | 1.0000 ±0.0000 |
| vdb100k | x64 | lloyd:10000:100 | 5 | 21.4 | -0.00% ±0.06 | 0.5081 ±0.0056 | 0.9345 ±0.0027 |
| vdb100k | x64 | lloyd:10000:25 | 5 | 13.8 | +0.00% ±0.06 | 0.5085 ±0.0040 | 0.9346 ±0.0034 |
| vdb100k | x64 | emeans:10000:5 | 5 | 7.0 | +0.12% ±0.06 | 0.5069 ±0.0019 | 0.9309 ±0.0022 |
| vdb100k | x64 | emeans:50000:5 | 5 | 20.3 | -1.52% ±0.05 | 0.5014 ±0.0030 | 0.9266 ±0.0012 |
| vdb100k | x64 | emeans:100000:5 | 5 | 35.4 | -1.88% ±0.03 | 0.5012 ±0.0032 | 0.9237 ±0.0019 |
| vdb100k | x64 | lloyd:100000:25 | 5 | 128.6 | -1.92% ±0.00 | 0.5068 ±0.0000 | 0.9283 ±0.0000 |

## Where the x64 error comes from

Score error = PQ score minus exact dot product, on each query's true top-10 ("NN") and on 10
random base points per query. 100K: 3 seeds, 1000 queries. 1M: 1-2 seeds, 500 queries.

100K:

| config | n | query MSE | recall@10 | rescored | NN bias | NN rmse | random bias | random rmse |
|---|---|---|---|---|---|---|---|---|
| lloyd:10000:100 | 3 | 0.17581 | 0.5161 | 0.9376 | -0.06499 | 0.06631 | +0.00006 | 0.01301 |
| emeans:10000:5 | 3 | 0.17608 | 0.5115 | 0.9323 | -0.06476 | 0.06611 | -0.00003 | 0.01294 |
| emeans:10000:5:norev | 3 | 0.17680 | 0.5052 | 0.9314 | -0.06507 | 0.06640 | -0.00016 | 0.01293 |
| emeans:10000:5:thr | 3 | 0.17616 | 0.5072 | 0.9319 | -0.06490 | 0.06624 | -0.00008 | 0.01289 |
| emeans:10000:20 | 3 | 0.17606 | 0.5021 | 0.9321 | -0.06397 | 0.06531 | -0.00007 | 0.01289 |
| emeans:100000:5 | 3 | 0.17255 | 0.5064 | 0.9257 | -0.06384 | 0.06518 | -0.00003 | 0.01279 |
| emeans:100000:5:norev | 3 | 0.17260 | 0.5069 | 0.9261 | -0.06380 | 0.06514 | -0.00009 | 0.01223 |

1M:

| config | n | query MSE | recall@10 | rescored | NN bias | NN rmse | random bias | random rmse |
|---|---|---|---|---|---|---|---|---|
| lloyd:10000:100 | 2 | 0.16888 | 0.5100 | 0.9215 | -0.08212 | 0.08375 | -0.00056 | 0.01372 |
| emeans:10000:5 | 2 | 0.16823 | 0.5091 | 0.9180 | -0.08086 | 0.08255 | -0.00030 | 0.01749 |
| emeans:100000:5 | 2 | 0.16396 | 0.5079 | 0.9094 | -0.08099 | 0.08267 | -0.00028 | 0.01720 |
| lloyd:100000:25 | 1 | 0.16428 | 0.5184 | 0.9122 | -0.08176 | 0.08345 | -0.00042 | 0.01358 |

The bias on true neighbours (-0.065 to -0.082) is 5x the random-point error and the same for every
variant, so it is a property of MSE-trained PQ on these embeddings, not of the k-means algorithm.
What separates Lloyd from e-means at x64 is not visible in these numbers. At 1M, e-means does show a
larger error on random points (0.017 vs 0.014), which could push non-neighbours into the top-100.
That is a lead, not a verified cause.

## Appendix: SIFT-1M, GloVe-100, dbpedia shards (first round)

Earlier harness run, 3 seeds, 1000 queries, 4 threads; dbpedia here is 100k rows from 3 parquet
shards of the HF dataset, with self-computed GT. Held-out MSE is on base rows outside the sample.

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

## Reproduce

Bench inputs are raw files in one directory: `base.f32` and `query.f32` (row-major f32),
`gt.u32` (top-k ids per query) and `meta.json` (`n`, `d`, `nq`, `metric`, `gt_k`). Convert the
vector-db-benchmark `vectors.npy` / `tests.jsonl` into that layout first.

```sh
cargo run --release --example pq_codebook_bench -- <dir> 16 4 5 lloyd:10000:100,emeans:10000:5 2000
cargo run --release --example pq_e2e_bench -- <dir> 4 4 2000 label
```

## Not covered

- x32 (8-dim chunks), other embedding models, GPU.
- Server test at 1M (indexing 1M x 1536-d several times on 4 cores was out of budget).
- `stop_condition_u8` fails with and without this change: in release mode, scalar encoding of 1M
  zero vectors finishes before the 100 ms stop signal fires.
