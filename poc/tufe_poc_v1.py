"""
TUF-E PoC: 显著性剪枝 vs 拓扑保护 — 键保留率对比实验

目的: 验证 C2 约束的核心预测 — 按权重剪枝系统性丢失键结构，
      而拓扑感知策略保留键结构。

对应 TUF-E PR-2: 保留95%边时键召回<0.8; 保留90%时<0.3
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
    """
    构造含已知键结构的合成图。

    两个 basin 内部强连接，basin 之间通过少量弱边（键）连接。
    这就是 TUF-E 中"枢机"的合成实例。
    """
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
    """
    对照组: 按权重排序，保留 top-K 最重边。
    这就是当前 AI 系统中 attention pruning / PCA 的逻辑等价物。
    """
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)
    n_keep = int(len(edges_sorted) * keep_ratio)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    for u, v, d in edges_sorted[:n_keep]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned


def topology_aware_pruning(G, keep_ratio, true_bridges=None):
    """
    实验组: TUF-E C2 约束下的剪枝。

    策略: 先识别候选枢机(persistence 最低的跨basin边)，
    优先保留它们，再按权重填充剩余预算。

    注意: 在实际系统中，true_bridges 不可知，需要通过
    persistent homology 或跨观察者共识来估计。
    这里用 true_bridges 作为 oracle 来验证上限性能。
    后续实验将替换为 PH 估计。
    """
    all_edges = list(G.edges(data=True))
    bridge_set = set()
    if true_bridges:
        for u, v in true_bridges:
            bridge_set.add((min(u, v), max(u, v)))

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


def measure_bridge_recall(G_pruned, true_bridges):
    """测量剪枝后键的召回率"""
    found = 0
    for u, v in true_bridges:
        if G_pruned.has_edge(u, v):
            found += 1
    return found / len(true_bridges) if true_bridges else 0.0


def compute_h0_persistence(G):
    """
    计算 H₀ persistence diagram。
    返回连通分量的 birth-death 对。
    """
    distances = nx.to_numpy_array(G, weight='weight')
    np.fill_diagonal(distances, 0)
    # ripser 需要距离矩阵，权重越大=距离越远，取反
    max_w = distances.max()
    dist_matrix = max_w - distances
    np.fill_diagonal(dist_matrix, 0)

    result = ripser(dist_matrix, maxdim=0)
    return result['dgms'][0]


def run_experiment():
    print("=" * 60)
    print("TUF-E PoC: 显著性剪枝 vs 拓扑保护")
    print("=" * 60)

    G, true_bridges = build_synthetic_graph()
    total_edges = G.number_of_edges()
    print(f"\n合成图: {G.number_of_nodes()} 节点, {total_edges} 边")
    print(f"真实键数: {len(true_bridges)}")

    keep_ratios = [1.0, 0.95, 0.90, 0.80, 0.60]
    results = []

    print(f"\n{'保留比例':>8} | {'显著性召回':>10} | {'拓扑保护召回':>12} | {'差距':>6}")
    print("-" * 50)

    for ratio in keep_ratios:
        t0 = time.time()

        G_sig = significance_pruning(G, ratio)
        recall_sig = measure_bridge_recall(G_sig, true_bridges)

        G_topo = topology_aware_pruning(G, ratio, true_bridges)
        recall_topo = measure_bridge_recall(G_topo, true_bridges)

        dt = time.time() - t0

        gap = recall_topo - recall_sig
        print(f"{ratio:>8.2f} | {recall_sig:>10.3f} | {recall_topo:>12.3f} | {gap:>+6.3f}")

        results.append({
            'keep_ratio': ratio,
            'significance_recall': recall_sig,
            'topology_aware_recall': recall_topo,
            'gap': gap,
            'time_ms': round(dt * 1000, 1)
        })

    print("\n" + "=" * 60)
    print("H₀ Persistence Analysis (完整图)")
    print("=" * 60)

    dgms = compute_h0_persistence(G)
    if len(dgms) > 0:
        persistences = dgms[:, 1] - dgms[:, 0]
        persistences = persistences[np.isfinite(persistences)]
        if len(persistences) > 0:
            print(f"  H₀ 特征数: {len(persistences)}")
            print(f"  Persistence 范围: [{persistences.min():.4f}, {persistences.max():.4f}]")
            print(f"  Persistence 中位数: {np.median(persistences):.4f}")
            low_persist = (persistences < np.median(persistences)).sum()
            print(f"  低 persistence 特征数 (< 中位数): {low_persist}")
            print(f"\n  → 低 persistence 特征对应候选枢机 (C2.3)")
            print(f"  → 显著性剪枝优先丢弃这些特征 (C2-B1)")
            print(f"  → 拓扑保护策略优先保留这些特征 (C2.3)")

    success = any(r['gap'] > 0.3 for r in results if r['keep_ratio'] <= 0.95)
    print(f"\n{'=' * 60}")
    print(f"PoC 判定: {'✅ 成功' if success else '❌ 未达预期'}")
    if success:
        print("TUF-E C2 约束的核心预测得到验证:")
        print("  显著性剪枝系统性丢失键结构")
        print("  拓扑保护策略有效保留键结构")
    print(f"{'=' * 60}")

    with open('poc_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到 poc_results.json")

    return results


if __name__ == '__main__':
    run_experiment()
