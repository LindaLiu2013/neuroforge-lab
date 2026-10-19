"""
EXP-003: 自适应参数确认 + 复杂度压力测试

Part A: EXP-002发现的最优参数在20种子上确认
Part B: 多basin/星型/层次拓扑下的算法表现
"""

import numpy as np
import networkx as nx
import json
import time
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def build_graph_baseline(seed=42):
    """V1: 2 basin, 5 bridges (同EXP-001)"""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_per_basin, n_basins, n_bridge_edges = 30, 2, 5
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    G.add_edge(offset + i, offset + j, weight=rng.uniform(0.7, 1.0))
    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin)) for b in range(n_basins)]
    for _ in range(n_bridge_edges):
        b1, b2 = rng.choice(n_basins, size=2, replace=False)
        u, v = rng.choice(basin_nodes[b1]), rng.choice(basin_nodes[b2])
        w = rng.uniform(0.05, 0.15)
        G.add_edge(u, v, weight=w)
        true_bridges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, 0.05)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridges


def build_graph_chain(seed=42):
    """V2: 4 basin chain, 10 bridges (basin_i <-> basin_{i+1})"""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_per_basin, n_basins = 20, 4
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    G.add_edge(offset + i, offset + j, weight=rng.uniform(0.7, 1.0))
    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin)) for b in range(n_basins)]
    # Chain: connect adjacent basins with 3 bridges each (except last pair gets 4 to total 10)
    bridge_counts = [3, 3, 4]  # 3 pairs * ~3 = ~9-10
    for pair_idx in range(n_basins - 1):
        b1, b2 = pair_idx, pair_idx + 1
        n_b = bridge_counts[pair_idx] if pair_idx < len(bridge_counts) else 3
        for _ in range(n_b):
            u, v = rng.choice(basin_nodes[b1]), rng.choice(basin_nodes[b2])
            w = rng.uniform(0.05, 0.15)
            G.add_edge(u, v, weight=w)
            true_bridges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, 0.05)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridges


def build_graph_star(seed=42):
    """V3: 4 basin star, center basin connects to all others"""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_per_basin, n_basins = 20, 4
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    G.add_edge(offset + i, offset + j, weight=rng.uniform(0.7, 1.0))
    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin)) for b in range(n_basins)]
    # Star: basin 0 is center, connects to basins 1,2,3 with 2 bridges each
    for b in range(1, n_basins):
        for _ in range(2):
            u, v = rng.choice(basin_nodes[0]), rng.choice(basin_nodes[b])
            w = rng.uniform(0.05, 0.15)
            G.add_edge(u, v, weight=w)
            true_bridges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, 0.05)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridges


def build_graph_hierarchical(seed=42):
    """V4: 3 basin with sub-basins (internal strong clusters within each basin)"""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_basins = 3
    n_sub = 2  # sub-basins per basin
    n_per_sub = 15  # nodes per sub-basin
    # Intra-sub-basin edges (very strong)
    for b in range(n_basins):
        for s in range(n_sub):
            offset = b * n_sub * n_per_sub + s * n_per_sub
            for i in range(n_per_sub):
                for j in range(i + 1, n_per_sub):
                    if rng.random() < 0.5:
                        G.add_edge(offset + i, offset + j, weight=rng.uniform(0.8, 1.0))
        # Inter-sub-basin edges within same basin (moderate)
        off0 = b * n_sub * n_per_sub
        off1 = off0 + n_per_sub
        for _ in range(8):
            u = rng.choice(range(off0, off0 + n_per_sub))
            v = rng.choice(range(off1, off1 + n_per_sub))
            G.add_edge(u, v, weight=rng.uniform(0.4, 0.6))
    # Inter-basin bridges (weak)
    basin_nodes = [list(range(b * n_sub * n_per_sub, (b + 1) * n_sub * n_per_sub))
                   for b in range(n_basins)]
    bridge_pairs = [(0, 1), (1, 2), (0, 2)]
    for b1, b2 in bridge_pairs:
        n_b = 3 if (b1, b2) != (0, 2) else 2  # total 8
        for _ in range(n_b):
            u, v = rng.choice(basin_nodes[b1]), rng.choice(basin_nodes[b2])
            w = rng.uniform(0.05, 0.15)
            G.add_edge(u, v, weight=w)
            true_bridges.append((u, v))
    for u, v, d in G.edges(data=True):
        d['weight'] += rng.normal(0, 0.05)
        d['weight'] = max(0.01, d['weight'])
    return G, true_bridges


def detect_candidate_pivots_param(G, alpha=0.2, weight_power=2.0):
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


def ph_aware_pruning_param(G, keep_ratio, alpha=0.2, weight_power=2.0, protect_k=5):
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


def significance_pruning(G, keep_ratio):
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)
    n_keep = int(len(edges_sorted) * keep_ratio)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in edges_sorted[:n_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def measure_bridge_recall(G_pruned, true_bridges):
    found = sum(1 for u, v in true_bridges if G_pruned.has_edge(u, v))
    return found / len(true_bridges) if true_bridges else 0.0


def run_experiment():
    print("=" * 70)
    print("EXP-003: 自适应参数确认 + 复杂度压力测试")
    print(f"开始时间: {datetime.now().isoformat()}")
    print("=" * 70)

    output = {'exp_id': 'EXP-003', 'timestamp': datetime.now().isoformat()}

    # === PART A ===
    print("\n" + "=" * 70)
    print("PART A: 最优参数确认 (alpha=0.2, k=5, p=2.0)")
    print("=" * 70)

    seeds_a = list(range(20))
    keep_ratios = [0.95, 0.90]
    part_a_results = []

    for seed in seeds_a:
        G, tb = build_graph_baseline(seed=seed)
        seed_data = {'seed': seed, 'n_edges': G.number_of_edges(), 'n_bridges': len(tb)}

        for ratio in keep_ratios:
            ratio_key = str(ratio)
            G_oracle = oracle_pruning(G, ratio, tb)
            recall_oracle = measure_bridge_recall(G_oracle, tb)
            G_ph = ph_aware_pruning_param(G, ratio, alpha=0.2, weight_power=2.0, protect_k=5)
            recall_ph = measure_bridge_recall(G_ph, tb)
            eff = recall_ph / recall_oracle if recall_oracle > 0 else 1.0
            seed_data[f'eff_{ratio_key}'] = round(eff, 4)
            seed_data[f'recall_ph_{ratio_key}'] = round(recall_ph, 4)

        part_a_results.append(seed_data)
        avg_eff = np.mean([seed_data[f'eff_{str(r)}'] for r in keep_ratios])
        print(f"  Seed {seed:>2}: edges={G.number_of_edges():>3}, "
              f"eff_95={seed_data['eff_0.95']:.3f}, eff_90={seed_data['eff_0.9']:.3f}")

    effs_95 = [d['eff_0.95'] for d in part_a_results]
    effs_90 = [d['eff_0.9'] for d in part_a_results]
    mean_eff_a = np.mean(effs_95 + effs_90)
    min_eff_a = np.min(effs_95 + effs_90)

    c1_pass = mean_eff_a >= 0.85
    c2_pass = min_eff_a >= 0.50

    print(f"\n  Part A 汇总:")
    print(f"    Mean efficiency: {mean_eff_a:.4f} (>= 0.85? {'PASS' if c1_pass else 'FAIL'})")
    print(f"    Min efficiency:  {min_eff_a:.4f} (>= 0.50? {'PASS' if c2_pass else 'FAIL'})")
    print(f"    Eff@0.95: mean={np.mean(effs_95):.4f}, std={np.std(effs_95):.4f}")
    print(f"    Eff@0.90: mean={np.mean(effs_90):.4f}, std={np.std(effs_90):.4f}")

    part_a_pass = c1_pass and c2_pass
    output['part_a'] = {
        'mean_efficiency': round(mean_eff_a, 4),
        'min_efficiency': round(min_eff_a, 4),
        'pass': bool(part_a_pass),
        'per_seed': part_a_results
    }

    # === PART B ===
    print("\n" + "=" * 70)
    print("PART B: 复杂度压力测试")
    print("=" * 70)

    variants = {
        'V1_baseline': build_graph_baseline,
        'V2_chain': build_graph_chain,
        'V3_star': build_graph_star,
        'V4_hierarchical': build_graph_hierarchical,
    }

    seeds_b = list(range(10))
    ratio_b = 0.95
    part_b_results = {}

    for vname, builder in variants.items():
        print(f"\n  --- {vname} ---")
        effs = []
        recalls_ph = []
        recalls_sig = []

        for seed in seeds_b:
            G, tb = builder(seed=seed)
            G_oracle = oracle_pruning(G, ratio_b, tb)
            recall_oracle = measure_bridge_recall(G_oracle, tb)
            G_ph = ph_aware_pruning_param(G, ratio_b, alpha=0.2, weight_power=2.0, protect_k=5)
            recall_ph = measure_bridge_recall(G_ph, tb)
            G_sig = significance_pruning(G, ratio_b)
            recall_sig = measure_bridge_recall(G_sig, tb)
            eff = recall_ph / recall_oracle if recall_oracle > 0 else 1.0
            effs.append(eff)
            recalls_ph.append(recall_ph)
            recalls_sig.append(recall_sig)

        mean_eff = np.mean(effs)
        mean_ph = np.mean(recalls_ph)
        mean_sig = np.mean(recalls_sig)
        passed = mean_eff >= 0.50

        part_b_results[vname] = {
            'mean_efficiency': round(mean_eff, 4),
            'mean_ph_recall': round(mean_ph, 4),
            'mean_sig_recall': round(mean_sig, 4),
            'std_efficiency': round(np.std(effs), 4),
            'pass': bool(passed),
            'per_seed_eff': [round(e, 4) for e in effs]
        }

        print(f"    Nodes~{G.number_of_nodes()}, Edges~{G.number_of_edges()}, Bridges={len(tb)}")
        print(f"    PH recall={mean_ph:.3f}, Sig recall={mean_sig:.3f}, "
              f"Eff={mean_eff:.3f} -> {'PASS' if passed else 'FAIL'}")

    output['part_b'] = part_b_results

    # === Overall ===
    part_b_all_pass = all(v['pass'] for v in part_b_results.values())
    overall_pass = part_a_pass and part_b_all_pass

    print(f"\n{'=' * 70}")
    print(f"EXP-003 总体判定: {'CONFIRMED' if overall_pass else 'PARTIAL'}")
    print(f"  Part A (最优参数): {'PASS' if part_a_pass else 'FAIL'}")
    for vname, vdata in part_b_results.items():
        print(f"  Part B {vname}: {'PASS' if vdata['pass'] else 'FAIL'} (eff={vdata['mean_efficiency']:.3f})")

    if not overall_pass:
        print("\n未通过的维度需要进一步分析，但不否定核心算法。")
        print("可能的改进方向: 自适应alpha、层次化社区检测、边权重归一化。")
    print(f"{'=' * 70}")

    # Counter-arguments
    print("""
反论点义务:
  1. Part A 的最优参数来自 EXP-002 的同一批数据(种子重叠)，存在循环验证风险
     → 缓解: Part B 使用不同拓扑，若也通过则降低过拟合嫌疑
  2. V2-V4 的合成复杂度仍远低于真实系统(无噪声异质性、无时序演化)
  3. Louvain seed=42 固定可能在某些拓扑下产生系统性偏差
  4. 桥边权重范围(0.05-0.15)在所有变体中相同，未测试权重重叠场景
""")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open('EXP-003_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    with open('audit/EXP-003_audit.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'event': 'experiment_complete',
                            'result': 'CONFIRMED' if overall_pass else 'PARTIAL',
                            'timestamp': datetime.now().isoformat()}, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到 EXP-003_results.json")
    return output


if __name__ == '__main__':
    run_experiment()
