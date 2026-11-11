import sys, os, json, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
v = np.load(f"{src}/vectors.npy", mmap_mode="r")
qs, gt, top1 = [], [], []
for line in open(f"{src}/tests.jsonl"):
    d = json.loads(line); qs.append(d["query"]); gt.append(d["closest_ids"][:10]); top1.append(d["closest_scores"][0])
q = np.asarray(qs, np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
# vectors.npy rows are already unit norm; write in chunks to bound memory
with open(f"{out}/base.f32", "wb") as f:
    for i in range(0, len(v), 100_000):
        b = np.array(v[i:i+100_000], np.float32); b /= np.linalg.norm(b, axis=1, keepdims=True); b.tofile(f)
q.tofile(f"{out}/query.f32"); np.asarray(gt, np.uint32).tofile(f"{out}/gt.u32")
json.dump(dict(n=len(v), d=v.shape[1], nq=len(q), metric="dot", gt_k=10), open(f"{out}/meta.json", "w"))
# sanity: GT agrees with exact dot product on a few queries
for i in range(3):
    s = np.asarray(v, np.float32) @ q[i] if len(v) <= 200_000 else None
    if s is not None: print("gt check", i, set(np.argsort(-s)[:10]) == set(gt[i]))
print(len(v), v.shape[1], len(q), "max top1 score", max(top1))
