"""
EXP-001: 多信号枢机检测鲁棒性测试 (预注册确认性实验)

冻结参数来自 tufe_poc_v2.py，不允许事后修改。
成功判据见 preregister/EXP-001_robustness.yaml
"""

import numpy as np
import networkx as nx
from ripser import ripser
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
        score = wr * br * (0.5 if cc else 1.0)
        scored_edges.append((u, v, d['weight'], score, cc, betw.get(key, 0)))

    scored_edges.sort(key=lambda x: x[3])
    return scored_edges


def ph_aware_pruning(G, keep_ratio, n_protect=None):
    scored_edges = detect_candidate_pivots(G)

    if n_protect is None:
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


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test (pure implementation, no scipy dependency)."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 1.0

    abs_diffs = sorted([(abs(d), i) for i, d in enumerate(diffs)])
    ranks = {}
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    W_plus = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    W_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    W = min(W_plus, W_minus)

    mu = n * (n + 1) / 4.0
    sigma = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    z = (W - mu) / sigma if sigma > 0 else 0

    # Two-tailed p-value approximation using normal distribution
    p = 2 * normal_cdf(z)
    return p


def normal_cdf(z):
    """Approximation of standard normal CDF."""
    import math
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p_val = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z)
    t = 1.0 / (1.0 + p_val * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def run_robustness_test():
    audit_log = []
    start_time = datetime.now().isoformat()

    print("=" * 70)
    print("EXP-001: 多信号枢机检测鲁棒性测试")
    print(f"开始时间: {start_time}")
    print("=" * 70)

    seeds = list(range(20))
    keep_ratios = [1.0, 0.95, 0.90, 0.80, 0.60]

    all_results = []
    seed_summaries = []

    for seed in seeds:
        t0 = time.time()
        G, true_bridges = build_synthetic_graph(seed=seed)

        seed_data = {
            'seed': seed,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_true_bridges': len(true_bridges),
            'keep_ratios': {}
        }

        for ratio in keep_ratios:
            G_sig = significance_pruning(G, ratio)
            recall_sig = measure_bridge_recall(G_sig, true_bridges)

            G_oracle = oracle_pruning(G, ratio, true_bridges)
            recall_oracle = measure_bridge_recall(G_oracle, true_bridges)

            G_ph = ph_aware_pruning(G, ratio)
            recall_ph = measure_bridge_recall(G_ph, true_bridges)

            efficiency = recall_ph / recall_oracle if recall_oracle > 0 else (1.0 if recall_ph == 0 else 0.0)

            seed_data['keep_ratios'][str(ratio)] = {
                'significance_recall': round(recall_sig, 4),
                'oracle_recall': round(recall_oracle, 4),
                'ph_auto_recall': round(recall_ph, 4),
                'efficiency': round(efficiency, 4)
            }

        dt = time.time() - t0
        seed_data['time_s'] = round(dt, 2)

        # Compute per-seed average efficiency (exclude ratio=1.0)
        effs = [seed_data['keep_ratios'][str(r)]['efficiency']
                for r in keep_ratios if r < 1.0]
        seed_data['avg_efficiency'] = round(np.mean(effs), 4)

        seed_summaries.append(seed_data)
        all_results.append(seed_data)

        print(f"  Seed {seed:>2}: nodes={G.number_of_nodes():>3}, "
              f"edges={G.number_of_edges():>3}, bridges={len(true_bridges)}, "
              f"avg_eff={seed_data['avg_efficiency']:.3f}, time={dt:.1f}s")

        audit_log.append({
            'event': 'seed_complete',
            'seed': seed,
            'avg_efficiency': seed_data['avg_efficiency'],
            'timestamp': datetime.now().isoformat()
        })

    # === Aggregate statistics ===
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    avg_efficiencies = [s['avg_efficiency'] for s in seed_summaries]
    mean_eff = np.mean(avg_efficiencies)
    std_eff = np.std(avg_efficiencies)
    min_eff = np.min(avg_efficiencies)
    max_eff = np.max(avg_efficiencies)
    median_eff = np.median(avg_efficiencies)

    print(f"\n平均效率 (across seeds, ratio<1.0):")
    print(f"  Mean:   {mean_eff:.4f}")
    print(f"  Median: {median_eff:.4f}")
    print(f"  Std:    {std_eff:.4f}")
    print(f"  Min:    {min_eff:.4f}")
    print(f"  Max:    {max_eff:.4f}")

    # Per-ratio breakdown
    print(f"\n按保留比例分解:")
    print(f"  {'Ratio':>6} | {'Sig Mean':>9} | {'Oracle Mean':>11} | {'PH Mean':>8} | {'Eff Mean':>9} | {'Eff Std':>8}")
    print(f"  " + "-" * 62)

    ratio_stats = {}
    for ratio in keep_ratios:
        sig_recs = [s['keep_ratios'][str(ratio)]['significance_recall'] for s in seed_summaries]
        orc_recs = [s['keep_ratios'][str(ratio)]['oracle_recall'] for s in seed_summaries]
        ph_recs = [s['keep_ratios'][str(ratio)]['ph_auto_recall'] for s in seed_summaries]
        effs = [s['keep_ratios'][str(ratio)]['efficiency'] for s in seed_summaries]

        ratio_stats[str(ratio)] = {
            'sig_mean': round(np.mean(sig_recs), 4),
            'oracle_mean': round(np.mean(orc_recs), 4),
            'ph_mean': round(np.mean(ph_recs), 4),
            'eff_mean': round(np.mean(effs), 4),
            'eff_std': round(np.std(effs), 4)
        }

        print(f"  {ratio:>6.2f} | {np.mean(sig_recs):>9.4f} | {np.mean(orc_recs):>11.4f} | "
              f"{np.mean(ph_recs):>8.4f} | {np.mean(effs):>9.4f} | {np.std(effs):>8.4f}")

    # === Success criteria evaluation ===
    print(f"\n{'=' * 70}")
    print("成功判据评估")
    print(f"{'=' * 70}")

    # Criterion 1: mean efficiency >= 0.70
    c1_pass = mean_eff >= 0.70
    print(f"\n[C1] 平均效率 >= 0.70: {mean_eff:.4f} -> {'PASS' if c1_pass else 'FAIL'}")

    # Criterion 2: worst seed >= 0.40
    c2_pass = min_eff >= 0.40
    print(f"[C2] 最差种子效率 >= 0.40: {min_eff:.4f} -> {'PASS' if c2_pass else 'FAIL'}")

    # Criterion 3: PH > significance (Wilcoxon)
    # Test at each ratio < 1.0
    c3_results = {}
    c3_all_pass = True
    for ratio in keep_ratios:
        if ratio >= 1.0:
            continue
        ph_recs = [s['keep_ratios'][str(ratio)]['ph_auto_recall'] for s in seed_summaries]
        sig_recs = [s['keep_ratios'][str(ratio)]['significance_recall'] for s in seed_summaries]
        p_val = wilcoxon_signed_rank(ph_recs, sig_recs)
        ph_gt_sig = np.mean(ph_recs) > np.mean(sig_recs)
        passed = ph_gt_sig and p_val < 0.05
        c3_results[str(ratio)] = {'p_value': round(p_val, 6), 'ph_gt_sig': ph_gt_sig, 'pass': passed}
        if not passed:
            c3_all_pass = False
        print(f"[C3] ratio={ratio:.2f}: PH>{'Sig' if ph_gt_sig else 'NOT>Sig'}, "
              f"p={p_val:.6f} -> {'PASS' if passed else 'FAIL'}")

    overall_pass = c1_pass and c2_pass and c3_all_pass

    print(f"\n{'=' * 70}")
    print(f"EXP-001 总体判定: {'CONFIRMED' if overall_pass else 'NOT CONFIRMED'}")
    if overall_pass:
        print("v2.2 探索性结果在 20 个种子上得到确认。")
        print("多信号融合枢机检测具有鲁棒性，非种子特异性。")
    else:
        failed = []
        if not c1_pass:
            failed.append(f"C1(mean_eff={mean_eff:.4f}<0.70)")
        if not c2_pass:
            failed.append(f"C2(min_eff={min_eff:.4f}<0.40)")
        if not c3_all_pass:
            failed.append("C3(Wilcoxon)")
        print(f"失败判据: {', '.join(failed)}")
        print("v2.2 结果未能完全复现，需进一步分析。")
    print(f"{'=' * 70}")

    # === Counter-argument obligation ===
    print(f"\n{'=' * 70}")
    print("反论点义务 (无论结果如何)")
    print(f"{'=' * 70}")
    print("""
替代解释:
  1. 合成图的桥边权重范围(0.05-0.15)与内部边(0.7-1.0)差距极大，
     使得三信号融合的检测任务比真实场景简单得多。
     真实数据中权重分布可能重叠，导致检测退化。

  2. Louvain 社区检测在双basin结构上几乎必然恢复正确分区，
     跨簇信号因此接近完美。多basin或层次化结构中，
     社区检测本身可能失败，导致跨簇信号失效。

生态效度威胁:
  - 合成图只有2个basin、5条键，拓扑极度简化
  - 权重是均匀分布+高斯噪声，真实数据的权重分布未知
  - 无向无权假设排除了方向性和异质性
  - 静态图忽略了真实系统的时序演化

置信度自评: 75%
  - 合成验证充分，但真实数据验证为零
  - 算法对参数(融合公式系数、n_protect启发式)的敏感性未系统测试
  - Louvain seed=42 固定可能导致社区检测结果偏差
""")

    # Save results
    output = {
        'exp_id': 'EXP-001',
        'timestamp': start_time,
        'completed': datetime.now().isoformat(),
        'parameters': {
            'seeds': seeds,
            'keep_ratios': keep_ratios,
            'n_per_basin': 30,
            'n_basins': 2,
            'n_bridge_edges': 5,
            'fusion_formula': 'wr * br * (0.5 if cc else 1.0)',
            'n_protect': 'max(n_communities * 3, 8)',
            'louvain_seed': 42
        },
        'aggregate': {
            'mean_efficiency': round(mean_eff, 4),
            'median_efficiency': round(median_eff, 4),
            'std_efficiency': round(std_eff, 4),
            'min_efficiency': round(min_eff, 4),
            'max_efficiency': round(max_eff, 4)
        },
        'ratio_stats': ratio_stats,
        'success_criteria': {
            'c1_mean_eff_ge_070': {'value': round(mean_eff, 4), 'pass': c1_pass},
            'c2_min_eff_ge_040': {'value': round(min_eff, 4), 'pass': c2_pass},
            'c3_wilcoxon': c3_results,
            'overall': 'CONFIRMED' if overall_pass else 'NOT CONFIRMED'
        },
        'per_seed': seed_summaries
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open('EXP-001_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # Audit log
    audit_log.append({
        'event': 'experiment_complete',
        'overall_result': 'CONFIRMED' if overall_pass else 'NOT CONFIRMED',
        'mean_efficiency': round(mean_eff, 4),
        'timestamp': datetime.now().isoformat()
    })
    with open('audit/EXP-001_audit.jsonl', 'w', encoding='utf-8') as f:
        for entry in audit_log:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到 EXP-001_results.json")
    print(f"审计日志已保存到 audit/EXP-001_audit.jsonl")

    return output


if __name__ == '__main__':
    run_robustness_test()
