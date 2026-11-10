import sys, os, glob, json, numpy as np, h5py, pyarrow.parquet as pq
D = sys.argv[1]; NQ = 1000
def save(name, base, query, gt, metric):
    out = os.path.join(D, name); os.makedirs(out, exist_ok=True)
    np.ascontiguousarray(base, np.float32).tofile(f"{out}/base.f32")
    np.ascontiguousarray(query, np.float32).tofile(f"{out}/query.f32")
    np.ascontiguousarray(gt[:, :100], np.uint32).tofile(f"{out}/gt.u32")
    json.dump(dict(n=len(base), d=base.shape[1], nq=len(query), metric=metric, gt_k=100), open(f"{out}/meta.json", "w"))
    print(name, base.shape, query.shape, metric)
def norm(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
with h5py.File(f"{D}/sift.hdf5") as f:
    save("sift", f["train"][:], f["test"][:NQ], f["neighbors"][:NQ], "l2")
with h5py.File(f"{D}/glove.hdf5") as f:
    save("glove", norm(f["train"][:]), norm(f["test"][:NQ]), f["neighbors"][:NQ], "dot")
files = sorted(glob.glob(f"{D}/dbp_*.parquet"))
emb = np.concatenate([np.stack(pq.read_table(p, columns=["openai"]).column("openai").to_numpy(zero_copy_only=False)) for p in files]).astype(np.float32)
emb = norm(emb); rng = np.random.default_rng(0); emb = emb[rng.permutation(len(emb))]
base, query = emb[:100_000], emb[100_000:100_000 + NQ]
scores = query @ base.T
gt = np.argsort(-scores, axis=1)[:, :100]
save("dbpedia", base, query, gt, "dot")
