import json, sys, collections, math, statistics as st
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}
def ci(x): return T95.get(len(x) - 1, 2.0) * st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0
g = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f): r = json.loads(l); g[(r['dataset'], r['bucket'], r['label'])].append(r)
print("| dataset | x | variant | n | encode s | recall@10 | rescored top-100 |")
print("|---|---|---|---|---|---|---|")
for k in sorted(g):
    rs = g[k]; f = lambda n: [r[n] for r in rs]
    print(f"| {k[0]} | x{k[1]*4} | {k[2]} | {len(rs)} | {st.mean(f('encode_s')):.1f} ±{ci(f('encode_s')):.1f} | {st.mean(f('recall10')):.4f} ±{ci(f('recall10')):.4f} | {st.mean(f('recall10_rescore100')):.4f} ±{ci(f('recall10_rescore100')):.4f} |")
