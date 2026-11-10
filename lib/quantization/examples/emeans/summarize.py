import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows: g[(r['dataset'], r['bucket'], r['threads'], r['config'])].append(r)
base = {}
for (ds, b, t, c), rs in g.items():
    if c == 'lloyd:10000:100': base[(ds, b, t)] = rs
def m(rs, k):
    if k == 'mse_heldout' and any(r[k] == 0 for r in rs): k = 'mse_query'
    return st.mean(r[k] for r in rs)
def sd(rs, k): return st.pstdev(r[k] for r in rs) if len(rs) > 1 else 0
print("| dataset | x | thr | config | train s | speedup | held-out MSE vs current | recall@10 | recall@10 rescored |")
print("|---|---|---|---|---|---|---|---|---|")
for (ds, b, t, c), rs in g.items():
    bl = base.get((ds, b, t))
    sp = m(bl, 'train_s') / m(rs, 'train_s') if bl else float('nan')
    dm = (m(rs, 'mse_heldout') / m(bl, 'mse_heldout') - 1) * 100 if bl else float('nan')
    print(f"| {ds} | x{b*4} | {t} | {c} | {m(rs,'train_s'):.2f} | {sp:.1f}x | {dm:+.2f}% | {m(rs,'recall10'):.4f} ±{sd(rs,'recall10'):.4f} | {m(rs,'recall10_rescore100'):.4f} |")
