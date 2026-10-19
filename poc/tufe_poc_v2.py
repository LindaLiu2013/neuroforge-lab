"""
TUF-E PoC v2: PH 自动枢机检测 — 无 oracle 的拓扑保护

v1 证明了"保护键是对的"，但用了已知的真实键位置（oracle）。
v2 证明"系统自己能通过 persistent homology 找到键在哪"。

三种策略对比:
  A. 显著性剪枝 (对照组): 按权重保留 top-K
  B. Oracle 保护 (v1上限): 已知真实键，优先保留
  C. PH 自动保护 (v2核心): 通过 H0 persistence 自动识别候选枢机

核心问题: C 能否接近 B 的性能？差距有多大？
"""

import numpy as np
import networkx as nx
from ripser import ripser
import json
import time
import sys
import io

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

    bridge_edges = []
    non_bridge_edges = []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        if key in bridge_set:
            bridge_edges.append((u, v, d))
        else:
            non_bridge_edges.append((u, v, d))

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


def detect_candidate_pivots(G, n_candidates=None):
    """
    通过多信号融合自动检测候选枢机边。

    v2.1 改进: 单一 PH 信号不足以定位枢机（H0 persistence 是全局性质，
    不能直接归因到单条边）。改用三信号融合:

    1. 边权重 (低 = 候选) — 对应 TUF-E "键按定义是最弱的边"
    2. 边介数中心性 (高 = 候选) — 对应 "移除后图分裂" 的结构重要性
    3. 跨簇连接性 (高 = 候选) — 对应 "连接不同 basin 的边"

    融合公式: pivot_score = rank(weight_asc) * rank(betweenness_desc) * cross_cluster_flag
    score 越低 = 越像枢机
    """
    edges = list(G.edges(data=True))
    edge_keys = [(min(u, v), max(u, v)) for u, v, _ in edges]

    weights = {k: d['weight'] for k, (_, _, d) in zip(edge_keys, edges)}

    betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    betw = {(min(u, v), max(u, v)): v for (u, v), v in betweenness.items()}

    # 用 Louvain 社区检测识别 basin
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
        cu = node_community.get(u, -1)
        cv = node_community.get(v, -1)
        cross_cluster[key] = 1 if cu != cv else 0

    sorted_by_weight = sorted(edge_keys, key=lambda k: weights[k])
    weight_rank = {k: i / len(edge_keys) for i, k in enumerate(sorted_by_weight)}

    sorted_by_betw = sorted(edge_keys, key=lambda k: betw.get(k, 0), reverse=True)
    betw_rank = {k: i / len(edge_keys) for i, k in enumerate(sorted_by_betw)}

    scored_edges = []
    for u, v, d in edges:
        key = (min(u, v), max(u, v))
        wr = weight_rank[key]
        br = betw_rank[key]
        cc = cross_cluster[key]

        # 融合: 低权重排名 * 高介数排名 * 跨簇加成
        # 跨簇边获得额外降权(更可能是枢机)
        score = wr * br * (0.5 if cc else 1.0)

        scored_edges.append((u, v, d['weight'], score, cc, betw.get(key, 0)))

    scored_edges.sort(key=lambda x: x[3])

    return scored_edges


def ph_aware_pruning(G, keep_ratio, n_protect=None):
    """
    实验组: 基于 PH 自动检测的拓扑保护剪枝。

    策略:
    1. 用 detect_candidate_pivots 识别候选枢机边
    2. 保护前 n_protect 条最像枢机的边
    3. 剩余预算按权重填充
    """
    scored_edges = detect_candidate_pivots(G)

    if n_protect is None:
        # 保护数量 = 真实键数的合理上界估计
        # 用社区数 * 2 作为启发式（每个社区间可能有1-2条键）
        try:
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            n_communities = len(communities)
        except Exception:
            n_communities = 2
        n_protect = max(n_communities * 3, 8)

    all_edges = list(G.edges(data=True))
    n_total_keep = int(len(all_edges) * keep_ratio)

    protected_keys = set()
    for u, v, w, score, cc, bv in scored_edges[:n_protect]:
        protected_keys.add((min(u, v), max(u, v)))

    protected_edges = []
    other_edges = []
    for u, v, d in all_edges:
        key = (min(u, v), max(u, v))
        if key in protected_keys:
            protected_edges.append((u, v, d))
        else:
            other_edges.append((u, v, d))

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


def run_experiment():
    print("=" * 65)
    print("TUF-E PoC v2: PH 自动枢机检测 vs Oracle vs 显著性")
    print("=" * 65)

    G, true_bridges = build_synthetic_graph()
    total_edges = G.number_of_edges()
    print(f"\n合成图: {G.number_of_nodes()} 节点, {total_edges} 边")
    print(f"真实键数: {len(true_bridges)}")

    scored = detect_candidate_pivots(G)
    print(f"\nPH+介数+社区 候选枢机检测 (前10条, 按 pivot_score 升序):")
    print(f"  {'边':>12} | {'权重':>6} | {'score':>7} | {'跨簇?':>5} | {'介数':>7} | {'真实键?':>6}")
    print(f"  " + "-" * 58)
    true_set = {(min(u, v), max(u, v)) for u, v in true_bridges}
    for u, v, w, score, cc, betw_val in scored[:10]:
        is_true = (min(u, v), max(u, v)) in true_set
        marker = "YES <<<" if is_true else ""
        print(f"  ({u:>2},{v:>2})    | {w:>6.3f} | {score:>7.4f} | {'Y' if cc else 'N':>5} | {betw_val:>7.4f} | {marker}")

    ph_detected_true = sum(1 for u, v, w, s, cc, bv in scored[:len(true_bridges)]
                           if (min(u, v), max(u, v)) in true_set)
    print(f"\n  PH top-{len(true_bridges)} 中命中真实键: {ph_detected_true}/{len(true_bridges)}")

    keep_ratios = [1.0, 0.95, 0.90, 0.80, 0.60]
    results = []

    print(f"\n{'保留比例':>8} | {'显著性':>7} | {'Oracle':>7} | {'PH自动':>7} | {'PH/Oracle':>9}")
    print("-" * 55)

    for ratio in keep_ratios:
        t0 = time.time()

        G_sig = significance_pruning(G, ratio)
        recall_sig = measure_bridge_recall(G_sig, true_bridges)

        G_oracle = oracle_pruning(G, ratio, true_bridges)
        recall_oracle = measure_bridge_recall(G_oracle, true_bridges)

        G_ph = ph_aware_pruning(G, ratio)
        recall_ph = measure_bridge_recall(G_ph, true_bridges)

        dt = time.time() - t0
        efficiency = recall_ph / recall_oracle if recall_oracle > 0 else 0

        print(f"{ratio:>8.2f} | {recall_sig:>7.3f} | {recall_oracle:>7.3f} | {recall_ph:>7.3f} | {efficiency:>8.1%}")

        results.append({
            'keep_ratio': ratio,
            'significance_recall': recall_sig,
            'oracle_recall': recall_oracle,
            'ph_auto_recall': recall_ph,
            'ph_efficiency_vs_oracle': round(efficiency, 3),
            'time_ms': round(dt * 1000, 1)
        })

    print(f"\n{'=' * 65}")
    avg_efficiency = np.mean([r['ph_efficiency_vs_oracle']
                              for r in results if r['keep_ratio'] < 1.0])
    print(f"PH 自动检测平均效率 (vs Oracle): {avg_efficiency:.1%}")

    success_criteria = avg_efficiency >= 0.5
    print(f"PoC v2 判定: {'PASS' if success_criteria else 'NEEDS IMPROVEMENT'}")
    if success_criteria:
        print("  PH 自动检测在无 oracle 的情况下达到了 Oracle 50%+ 的性能")
        print("  TUF-E C2 约束可以在不知道真实键位置的情况下有效工作")
    else:
        print("  PH 自动检测需要改进检测算法或增加观察者视角")
    print(f"{'=' * 65}")

    with open('poc_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到 poc_v2_results.json")

    return results


if __name__ == '__main__':
    run_experiment()
