import json, sys, collections, statistics as st
g = collections.defaultdict(list)
for l in open(sys.argv[1]): r = json.loads(l); g[r['config']].append(r)
print("| config | n | query MSE | recall@10 | rescored | NN bias | NN rmse | random bias | random rmse |")
print("|---|---|---|---|---|---|---|---|---|")
for c, rs in g.items():
    m = lambda k: st.mean(r[k] for r in rs)
    print(f"| {c} | {len(rs)} | {m('mse_query'):.5f} | {m('recall10'):.4f} | {m('recall10_rescore100'):.4f} | {m('nn_bias'):+.5f} | {m('nn_rmse'):.5f} | {m('rnd_bias'):+.5f} | {m('rnd_rmse'):.5f} |")
