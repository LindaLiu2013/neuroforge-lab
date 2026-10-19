"""
EXP-002: 参数敏感性分析 (预注册确认性实验)

测试多信号融合检测对三个关键参数的鲁棒性。
冻结5个代表性种子，固定 keep_ratio=0.95。
"""

import numpy as np
import networkx as nx
import json
import time
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def build_synthetic_graph(n_per_basin=30, n_basins=2, n_bridge_edges=5,
                          intra_weight_range=(0.7, 1.0),
                          bridge_weight_range=(0.05, 0.15),
                          noise_level=0.05, seed=42):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridge_edges = []
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    w = rng.uniform(*intra_weight_range)
                    G.add_edge(offset + i, offset + j, weight=w)
    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin))
                   for b in range(n_basins)]
    for _ in range(n_bridge_edges):
        b1, b2 = rng.choice(n_basins, size=2, replace=False)
        u = rng.choice(basin_nodes[b1])
        v = rng.choice(basin_nodes[b2])
        w = rng.uniform(*bridge_weight_range)
        G.add_edge(u, v, weight=w)
        true_bridge_edges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, noise_level)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridge_edges


def significance_pruning(G, keep_ratio):
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)
    n_keep = int(len(edges_sorted) * keep_ratio)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in edges_sorted[:n_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def oracle_pruning(G, keep_ratio, true_bridges):
    all_edges = list(G.edges(data=True))
    bridge_set = {(min(u, v), max(u, v)) for u, v in true_bridges}
    bridge_edges, non_bridge_edges = [], []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        (bridge_edges if key in bridge_set else non_bridge_edges).append((u, v, d))
    n_total_keep = int(len(all_edges) * keep_ratio)
    n_bridge_keep = min(len(bridge_edges), n_total_keep)
    n_other_keep = n_total_keep - n_bridge_keep
    non_bridge_sorted = sorted(non_bridge_edges, key=lambda e: e[2]['weight'], reverse=True)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in bridge_edges[:n_bridge_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    for u, v, d in non_bridge_sorted[:n_other_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def detect_candidate_pivots_param(G, alpha=0.5, weight_power=1.0):
    """参数化版本的枢机检测。alpha=跨簇惩罚, weight_power=权重排名指数。"""
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
    return scored_edges


def ph_aware_pruning_param(G, keep_ratio, alpha=0.5, weight_power=1.0, protect_k=3):
    scored_edges = detect_candidate_pivots_param(G, alpha=alpha, weight_power=weight_power)
    try:
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
        n_communities = len(communities)
    except Exception:
        n_communities = 2
    n_protect = max(n_communities * protect_k, 8)

    all_edges = list(G.edges(data=True))
    n_total_keep = int(len(all_edges) * keep_ratio)
    protected_keys = set()
    for u, v, w, score, cc, bv in scored_edges[:n_protect]:
        protected_keys.add((min(u, v), max(u, v)))

    protected_edges, other_edges = [], []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        (protected_edges if key in protected_keys else other_edges).append((u, v, d))

    n_protected_keep = min(len(protected_edges), n_total_keep)
    n_other_keep = n_total_keep - n_protected_keep
    other_sorted = sorted(other_edges, key=lambda e: e[2]['weight'], reverse=True)

    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in protected_edges[:n_protected_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    for u, v, d in other_sorted[:n_other_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def measure_bridge_recall(G_pruned, true_bridges):
    found = sum(1 for u, v in true_bridges if G_pruned.has_edge(u, v))
    return found / len(true_bridges) if true_bridges else 0.0


def run_sensitivity_analysis():
    print("=" * 70)
    print("EXP-002: 参数敏感性分析")
    print(f"开始时间: {datetime.now().isoformat()}")
    print("=" * 70)

    seeds = [0, 4, 9, 14, 19]
    keep_ratio = 0.95

    # Pre-build graphs
    graphs = {}
    for seed in seeds:
        G, true_bridges = build_synthetic_graph(seed=seed)
        graphs[seed] = (G, true_bridges)

    # Baseline oracle recall per seed
    oracle_recalls = {}
    for seed in seeds:
        G, tb = graphs[seed]
        G_oracle = oracle_pruning(G, keep_ratio, tb)
        oracle_recalls[seed] = measure_bridge_recall(G_oracle, tb)

    results = {'d1_alpha': {}, 'd2_protect_k': {}, 'd3_weight_power': {}}

    # === D1: Alpha scan ===
    print("\n--- D1: 跨簇惩罚系数 alpha ---")
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"  {'alpha':>6} | {'Mean Eff':>9} | {'Min Eff':>8} | {'Max Eff':>8}")
    print(f"  " + "-" * 40)

    d1_data = []
    for alpha in alphas:
        effs = []
        for seed in seeds:
            G, tb = graphs[seed]
            G_ph = ph_aware_pruning_param(G, keep_ratio, alpha=alpha)
            recall_ph = measure_bridge_recall(G_ph, tb)
            eff = recall_ph / oracle_recalls[seed] if oracle_recalls[seed] > 0 else 1.0
            effs.append(eff)
        mean_e, min_e, max_e = np.mean(effs), np.min(effs), np.max(effs)
        d1_data.append({'alpha': alpha, 'mean': round(mean_e, 4), 'min': round(min_e, 4),
                        'max': round(max_e, 4), 'per_seed': [round(e, 4) for e in effs]})
        print(f"  {alpha:>6.1f} | {mean_e:>9.4f} | {min_e:>8.4f} | {max_e:>8.4f}")

    results['d1_alpha'] = d1_data

    # D1 stability check: range within [0.2, 0.7]
    d1_stable_range = [d for d in d1_data if 0.2 <= d['alpha'] <= 0.7]
    d1_effs_stable = [d['mean'] for d in d1_stable_range]
    d1_spread = max(d1_effs_stable) - min(d1_effs_stable) if d1_effs_stable else 999
    d1_pass = d1_spread < 0.15
    print(f"\n  D1 稳定性 [0.2-0.7]: spread={d1_spread:.4f} -> {'PASS' if d1_pass else 'FAIL'} (< 0.15)")

    # === D2: Protect K scan ===
    print("\n--- D2: n_protect 倍数因子 k ---")
    ks = [1, 2, 3, 5, 8]
    print(f"  {'k':>6} | {'Mean Eff':>9} | {'Min Eff':>8} | {'Max Eff':>8}")
    print(f"  " + "-" * 40)

    d2_data = []
    for k in ks:
        effs = []
        for seed in seeds:
            G, tb = graphs[seed]
            G_ph = ph_aware_pruning_param(G, keep_ratio, protect_k=k)
            recall_ph = measure_bridge_recall(G_ph, tb)
            eff = recall_ph / oracle_recalls[seed] if oracle_recalls[seed] > 0 else 1.0
            effs.append(eff)
        mean_e, min_e, max_e = np.mean(effs), np.min(effs), np.max(effs)
        d2_data.append({'k': k, 'mean': round(mean_e, 4), 'min': round(min_e, 4),
                        'max': round(max_e, 4), 'per_seed': [round(e, 4) for e in effs]})
        print(f"  {k:>6d} | {mean_e:>9.4f} | {min_e:>8.4f} | {max_e:>8.4f}")

    results['d2_protect_k'] = d2_data

    d2_stable_range = [d for d in d2_data if 2 <= d['k'] <= 5]
    d2_effs_stable = [d['mean'] for d in d2_stable_range]
    d2_spread = max(d2_effs_stable) - min(d2_effs_stable) if d2_effs_stable else 999
    d2_pass = d2_spread < 0.15
    print(f"\n  D2 稳定性 [k=2-5]: spread={d2_spread:.4f} -> {'PASS' if d2_pass else 'FAIL'} (< 0.15)")

    # === D3: Weight power scan ===
    print("\n--- D3: 权重排名指数 p ---")
    powers = [0.5, 1.0, 2.0]
    print(f"  {'p':>6} | {'Mean Eff':>9} | {'Min Eff':>8} | {'Max Eff':>8}")
    print(f"  " + "-" * 40)

    d3_data = []
    for p in powers:
        effs = []
        for seed in seeds:
            G, tb = graphs[seed]
            G_ph = ph_aware_pruning_param(G, keep_ratio, weight_power=p)
            recall_ph = measure_bridge_recall(G_ph, tb)
            eff = recall_ph / oracle_recalls[seed] if oracle_recalls[seed] > 0 else 1.0
            effs.append(eff)
        mean_e, min_e, max_e = np.mean(effs), np.min(effs), np.max(effs)
        d3_data.append({'p': p, 'mean': round(mean_e, 4), 'min': round(min_e, 4),
                        'max': round(max_e, 4), 'per_seed': [round(e, 4) for e in effs]})
        print(f"  {p:>6.1f} | {mean_e:>9.4f} | {min_e:>8.4f} | {max_e:>8.4f}")

    results['d3_weight_power'] = d3_data

    d3_effs_all = [d['mean'] for d in d3_data]
    d3_spread = max(d3_effs_all) - min(d3_effs_all)
    d3_pass = d3_spread < 0.20
    print(f"\n  D3 稳定性 [p=0.5-2.0]: spread={d3_spread:.4f} -> {'PASS' if d3_pass else 'FAIL'} (< 0.20)")

    # === Summary ===
    overall_pass = d1_pass and d2_pass and d3_pass
    print(f"\n{'=' * 70}")
    print(f"EXP-002 总体判定: {'STABLE' if overall_pass else 'SENSITIVE'}")
    print(f"  D1 (alpha):  {'PASS' if d1_pass else 'FAIL'} (spread={d1_spread:.4f})")
    print(f"  D2 (k):      {'PASS' if d2_pass else 'FAIL'} (spread={d2_spread:.4f})")
    print(f"  D3 (power):  {'PASS' if d3_pass else 'FAIL'} (spread={d3_spread:.4f})")

    if overall_pass:
        print("\n算法在合理参数范围内稳定，EXP-001 结果可信度高。")
    else:
        failed = []
        if not d1_pass: failed.append(f"D1(spread={d1_spread:.4f})")
        if not d2_pass: failed.append(f"D2(spread={d2_spread:.4f})")
        if not d3_pass: failed.append(f"D3(spread={d3_spread:.4f})")
        print(f"\n敏感维度: {', '.join(failed)}")
        print("需要更精细的参数选择策略或自适应机制。")
    print(f"{'=' * 70}")

    # Counter-arguments
    print("""
反论点义务:
  1. 离散采样可能错过最优参数区域；连续优化可能找到显著更好的点
  2. 5个种子的样本量不足以估计参数-种子交互效应
  3. 只测了 keep_ratio=0.95；其他比例下敏感性可能不同
  4. 参数间可能存在交互效应（本实验未测试二维网格）
""")

    # Save
    output = {
        'exp_id': 'EXP-002',
        'timestamp': datetime.now().isoformat(),
        'parameters': {'seeds': seeds, 'keep_ratio': keep_ratio},
        'results': results,
        'success_criteria': {
            'd1_pass': bool(d1_pass), 'd1_spread': round(d1_spread, 4),
            'd2_pass': bool(d2_pass), 'd2_spread': round(d2_spread, 4),
            'd3_pass': bool(d3_pass), 'd3_spread': round(d3_spread, 4),
            'overall': 'STABLE' if overall_pass else 'SENSITIVE'
        }
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open('EXP-002_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    with open('audit/EXP-002_audit.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'event': 'experiment_complete', 'result': 'STABLE' if overall_pass else 'SENSITIVE',
                            'timestamp': datetime.now().isoformat()}, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到 EXP-002_results.json")
    return output


if __name__ == '__main__':
    run_sensitivity_analysis()
