"""
EXP-004: 权重重叠压力测试

系统缩小桥边/内部边权重差距，测量算法退化曲线。
目的不是通过，而是理解失效边界。
"""

import numpy as np
import networkx as nx
import json
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def build_graph_with_sep(seed, bridge_range, intra_range):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_per_basin, n_basins, n_bridge_edges = 30, 2, 5
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    G.add_edge(offset + i, offset + j, weight=rng.uniform(*intra_range))
    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin)) for b in range(n_basins)]
    for _ in range(n_bridge_edges):
        b1, b2 = rng.choice(n_basins, size=2, replace=False)
        u, v = rng.choice(basin_nodes[b1]), rng.choice(basin_nodes[b2])
        w = rng.uniform(*bridge_range)
        G.add_edge(u, v, weight=w)
        true_bridges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, 0.05)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridges


def detect_and_prune(G, keep_ratio, alpha=0.2, weight_power=2.0, protect_k=5):
    edges = list(G.edges(data=True))
    edge_keys = [(min(u, v), max(u, v)) for u, v, _ in edges]
    weights = {k: d['weight'] for k, (_, _, d) in zip(edge_keys, edges)}
    betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    betw = {(min(u, v), max(u, v)): v for (u, v), v in betweenness.items()}
    try:
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
    node_community = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_community[node] = cid
    cross_cluster = {}
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        cu, cv = node_community.get(u, -1), node_community.get(v, -1)
        cross_cluster[key] = 1 if cu != cv else 0
    sorted_by_weight = sorted(edge_keys, key=lambda k: weights[k])
    weight_rank = {k: i / len(edge_keys) for i, k in enumerate(sorted_by_weight)}
    sorted_by_betw = sorted(edge_keys, key=lambda k: betw.get(k, 0), reverse=True)
    betw_rank = {k: i / len(edge_keys) for i, k in enumerate(sorted_by_betw)}
    scored_edges = []
    for u, v, d in edges:
        key = (min(u, v), max(u, v))
        wr = weight_rank[key] ** weight_power
        br = betw_rank[key]
        cc = cross_cluster[key]
        score = wr * br * (alpha if cc else 1.0)
        scored_edges.append((u, v, d['weight'], score, cc, betw.get(key, 0)))
    scored_edges.sort(key=lambda x: x[3])

    n_protect = max(len(communities) * protect_k, 8)
    all_edges = list(G.edges(data=True))
    n_total_keep = int(len(all_edges) * keep_ratio)
    protected_keys = set()
    for u, v, w, score, cc, bv in scored_edges[:n_protect]:
        protected_keys.add((min(u, v), max(u, v)))
    protected_edges, other_edges = [], []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        (protected_edges if key in protected_keys else other_edges).append((u, v, d))
    n_pk = min(len(protected_edges), n_total_keep)
    n_ok = n_total_keep - n_pk
    other_sorted = sorted(other_edges, key=lambda e: e[2]['weight'], reverse=True)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in protected_edges[:n_pk]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    for u, v, d in other_sorted[:n_ok]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def oracle_pruning(G, keep_ratio, true_bridges):
    all_edges = list(G.edges(data=True))
    bridge_set = {(min(u, v), max(u, v)) for u, v in true_bridges}
    be, nbe = [], []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        (be if key in bridge_set else nbe).append((u, v, d))
    ntk = int(len(all_edges) * keep_ratio)
    nbk = min(len(be), ntk)
    nok = ntk - nbk
    ns = sorted(nbe, key=lambda e: e[2]['weight'], reverse=True)
    Gp = nx.Graph()
    Gp.add_nodes_from(G.nodes())
    for u, v, d in be[:nbk]: Gp.add_edge(u, v, weight=d['weight'])
    for u, v, d in ns[:nok]: Gp.add_edge(u, v, weight=d['weight'])
    return Gp


def sig_pruning(G, keep_ratio):
    es = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)
    nk = int(len(es) * keep_ratio)
    Gp = nx.Graph()
    Gp.add_nodes_from(G.nodes())
    for u, v, d in es[:nk]: Gp.add_edge(u, v, weight=d['weight'])
    return Gp


def recall(Gp, tb):
    return sum(1 for u, v in tb if Gp.has_edge(u, v)) / len(tb) if tb else 0.0


def run():
    print("=" * 70)
    print("EXP-004: 权重重叠压力测试")
    print(f"开始时间: {datetime.now().isoformat()}")
    print("=" * 70)

    conditions = {
        'strong':  {'bridge': (0.05, 0.15), 'intra': (0.70, 1.00), 'sep': 0.21},
        'medium':  {'bridge': (0.10, 0.25), 'intra': (0.60, 0.90), 'sep': 0.42},
        'weak':    {'bridge': (0.15, 0.35), 'intra': (0.50, 0.80), 'sep': 0.70},
        'overlap': {'bridge': (0.20, 0.45), 'intra': (0.45, 0.70), 'sep': 1.00},
    }

    seeds = list(range(10))
    keep_ratio = 0.95
    results = {}

    print(f"\n{'条件':>8} | {'sep':>5} | {'PH Eff':>7} | {'PH Rec':>7} | {'Sig Rec':>7} | {'Oracle Rec':>10}")
    print("-" * 55)

    for cname, cparams in conditions.items():
        ph_effs, ph_recs, sig_recs, orc_recs = [], [], [], []

        for seed in seeds:
            G, tb = build_graph_with_sep(seed, cparams['bridge'], cparams['intra'])

            G_orc = oracle_pruning(G, keep_ratio, tb)
            r_orc = recall(G_orc, tb)

            G_ph = detect_and_prune(G, keep_ratio)
            r_ph = recall(G_ph, tb)

            G_sig = sig_pruning(G, keep_ratio)
            r_sig = recall(G_sig, tb)

            eff = r_ph / r_orc if r_orc > 0 else 1.0
            ph_effs.append(eff)
            ph_recs.append(r_ph)
            sig_recs.append(r_sig)
            orc_recs.append(r_orc)

        mean_eff = np.mean(ph_effs)
        mean_ph = np.mean(ph_recs)
        mean_sig = np.mean(sig_recs)
        mean_orc = np.mean(orc_recs)

        results[cname] = {
            'sep_ratio': cparams['sep'],
            'ph_efficiency_mean': round(mean_eff, 4),
            'ph_efficiency_std': round(np.std(ph_effs), 4),
            'ph_recall_mean': round(mean_ph, 4),
            'sig_recall_mean': round(mean_sig, 4),
            'oracle_recall_mean': round(mean_orc, 4),
            'per_seed_ph_eff': [round(e, 4) for e in ph_effs],
            'per_seed_ph_rec': [round(r, 4) for r in ph_recs],
            'per_seed_sig_rec': [round(r, 4) for r in sig_recs],
        }

        print(f"{cname:>8} | {cparams['sep']:>5.2f} | {mean_eff:>7.3f} | "
              f"{mean_ph:>7.3f} | {mean_sig:>7.3f} | {mean_orc:>10.3f}")

    # Degradation analysis
    print(f"\n{'=' * 70}")
    print("退化分析")
    print(f"{'=' * 70}")

    seps = [results[c]['sep_ratio'] for c in conditions]
    effs = [results[c]['ph_efficiency_mean'] for c in conditions]

    print(f"\n退化曲线:")
    for cname in conditions:
        r = results[cname]
        bar = '#' * int(r['ph_efficiency_mean'] * 30)
        print(f"  sep={r['sep_ratio']:.2f}: {bar} ({r['ph_efficiency_mean']:.3f})")

    # Find critical point (eff drops below 0.5)
    critical_sep = None
    for cname in conditions:
        if results[cname]['ph_efficiency_mean'] < 0.5:
            critical_sep = results[cname]['sep_ratio']
            break
    if critical_sep:
        print(f"\n临界点 (eff < 0.5): sep_ratio ≈ {critical_sep:.2f}")
    else:
        print(f"\n未观察到临界点 — 即使在完全重叠下效率仍 >= 0.5")

    # Signal decomposition insight
    print(f"\n信号贡献分析:")
    print(f"  当权重信号退化(sep↑)时，介数+社区信号是否补偿？")
    for cname in conditions:
        r = results[cname]
        gap = r['ph_recall_mean'] - r['sig_recall_mean']
        print(f"  {cname:>8}: PH-Sig gap = {gap:+.3f} "
              f"(PH={r['ph_recall_mean']:.3f}, Sig={r['sig_recall_mean']:.3f})")

    # Counter-arguments
    print(f"\n{'=' * 70}")
    print("反论点义务")
    print(f"{'=' * 70}")
    print("""
  1. 噪声水平固定(0.05)，但真实噪声可能与权重尺度相关
  2. 只测了2-basin结构；多basin下跨簇信号的可靠性可能不同
  3. overlap条件下oracle本身也可能失效(键不可区分时oracle也靠运气)
  4. 退化曲线的精确形状需要更密的sep采样
""")

    # Save
    output = {
        'exp_id': 'EXP-004',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'seeds': seeds, 'keep_ratio': keep_ratio,
                       'detection_params': {'alpha': 0.2, 'weight_power': 2.0, 'protect_k': 5}},
        'conditions': {k: {'bridge': v['bridge'], 'intra': v['intra'], 'sep': v['sep']}
                       for k, v in conditions.items()},
        'results': results,
        'critical_sep': critical_sep
    }

    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open('EXP-004_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NE)

    with open('audit/EXP-004_audit.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'event': 'experiment_complete',
                            'critical_sep': critical_sep,
                            'timestamp': datetime.now().isoformat()}, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到 EXP-004_results.json")
    return output


if __name__ == '__main__':
    run()
