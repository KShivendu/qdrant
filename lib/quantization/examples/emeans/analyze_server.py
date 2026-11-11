import json, glob, os, re, sys, collections, math, statistics as st
S = sys.argv[1]
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 8: 2.31, 10: 2.23}
def ci(xs):
    if len(xs) < 2: return 0.0
    return T95.get(len(xs) - 1, 2.0) * st.stdev(xs) / math.sqrt(len(xs))
def welch(a, b):
    d = st.mean(b) - st.mean(a)
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    dfree = se**4 / ((st.variance(a)/len(a))**2/(len(a)-1) + (st.variance(b)/len(b))**2/(len(b)-1)) if se > 0 else 1
    return d, T95.get(max(1, round(dfree)), 2.0) * se
idx = collections.defaultdict(list); prec = collections.defaultdict(list); imp = collections.defaultdict(list)
for run in sorted(glob.glob(f"{S}/server/run*")):
    v = run.rsplit("-", 1)[1]
    # precise indexing: from points_count reaching 100000 to first green with all indexed, per lifecycle
    ev = [json.loads(l) for l in open(f"{run}/poll.jsonl") if '"status"' in l]
    t_full = None; got = []
    for e in ev:
        status, pts, indexed, _ = e["status"]
        if pts is not None and pts < 100000: t_full = None
        if pts == 100000 and t_full is None: t_full = e["t"]
        if status == "green" and pts == 100000 and indexed == 100000 and t_full is not None:
            got.append(e["t"] - t_full); t_full = -1  # mark lifecycle done
    for f in glob.glob(f"{run}/*.json"):
        d = json.load(open(f)); name = os.path.basename(f); comp = re.search(r"pq-(x\d+)", name).group(1)
        if "-upload-" in name:
            imp[(comp, v)].append(d["results"]["total_time"])
        else:
            c = d["params"]["config"]; q = c["quantization"]
            key = (comp, c["hnsw_ef"], q.get("rescore"), q.get("oversampling"))
            prec[key + (v,)].append(d["results"]["mean_precisions"])
    for comp, t in zip(("x16", "x64"), got): idx[(comp, v)].append(t)
print("## Indexing time after upload (s), precise poll")
for comp in ("x16", "x64"):
    a, b = idx[(comp, "old")], idx[(comp, "new")]
    d, h = welch(a, b) if len(a) > 1 and len(b) > 1 else (float("nan"), 0)
    print(f"{comp}: old {st.mean(a):.1f} ±{ci(a):.1f} (n={len(a)}), new {st.mean(b):.1f} ±{ci(b):.1f} (n={len(b)}), diff {d:+.1f} ±{h:.1f}  runs old={[round(x) for x in a]} new={[round(x) for x in b]}")
print("## vector-db-benchmark total import time (s, 5 s poll granularity)")
for comp in ("x16", "x64"):
    a, b = imp[(comp, "old")], imp[(comp, "new")]
    print(f"{comp}: old {st.mean(a):.1f} ±{ci(a):.1f}, new {st.mean(b):.1f} ±{ci(b):.1f}")
print("## precision@10 (5000 queries), mean ±95% CI over runs; diff = new - old")
print("| x | hnsw_ef | rescore | oversampling | old | new | diff |")
print("|---|---|---|---|---|---|---|")
keys = sorted({k[:4] for k in prec}, key=lambda k: (k[0], k[1], bool(k[2]), k[3] or 0))
for k in keys:
    a, b = prec[k + ("old",)], prec[k + ("new",)]
    if not a or not b: continue
    d, h = welch(a, b) if len(a) > 1 and len(b) > 1 else (st.mean(b) - st.mean(a), 0)
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3] or '-'} | {st.mean(a):.4f} ±{ci(a):.4f} | {st.mean(b):.4f} ±{ci(b):.4f} | {d:+.4f} ±{h:.4f} |")
