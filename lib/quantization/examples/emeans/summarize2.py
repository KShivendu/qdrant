import json, sys, collections, math, statistics as st
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}
def ci(x): return T95.get(len(x) - 1, 2.0) * st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows: g[(r['dataset'], r['bucket'], r['config'])].append(r)
def mse(r): return r['mse_query'] if r['dataset'].startswith('vdb') else (r['mse_heldout'] if r['mse_heldout'] > 0 else r['mse_query'])
print("| dataset | x | config | n | train s | held-out MSE vs current (queries for vdb*) | recall@10 | recall@10 rescored top-100 |")
print("|---|---|---|---|---|---|---|---|")
for (ds, b, c), rs in g.items():
    base = g[(ds, b, 'lloyd:10000:100')]
    m0 = st.mean(mse(r) for r in base)
    rel = [(mse(r) / m0 - 1) * 100 for r in rs]
    r10 = [r['recall10'] for r in rs]; rr = [r['recall10_rescore100'] for r in rs]; t = [r['train_s'] for r in rs]
    print(f"| {ds} | x{b*4} | {c} | {len(rs)} | {st.mean(t):.1f} | {st.mean(rel):+.2f}% ±{ci(rel):.2f} | {st.mean(r10):.4f} ±{ci(r10):.4f} | {st.mean(rr):.4f} ±{ci(rr):.4f} |")
