"""
TUF-E Tier 1 Prototype: 四模块集成验证

模块:
  1. TopologyProtectionEngine - 拓扑保护引擎 (C2, P3)
  2. MultiProjectionConsensus - 多投影共识模块 (C3)
  3. EnergyBudgetTracker - 能量预算追踪器 (P1, P2)
  4. LayerSeparationFramework - 层级分离框架 (C5)

目的: 验证四个模块可以在同一系统中协同工作，
      对同一数据执行完整的 TUF-E 约束流水线。
"""

import numpy as np
import networkx as nx
from ripser import ripser
import psutil
import json
import time
import os
import sys
import io
import multiprocessing as mp
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# MODULE 1: Topology Protection Engine (C2, P3)
# ============================================================
class TopologyProtectionEngine:
    """
    拓扑保护引擎。
    - 计算 persistent homology (H0)
    - 多信号融合枢机检测
    - C2-B1~B4 禁止操作检查
    """

    def __init__(self, alpha=0.2, weight_power=2.0, protect_k=5):
        self.alpha = alpha
        self.weight_power = weight_power
        self.protect_k = protect_k

    def compute_persistence(self, G):
        distances = nx.to_numpy_array(G, weight='weight')
        np.fill_diagonal(distances, 0)
        max_w = distances.max() if distances.max() > 0 else 1.0
        dist_matrix = max_w - distances
        np.fill_diagonal(dist_matrix, 0)
        result = ripser(dist_matrix, maxdim=0)
        return result['dgms'][0]

    def detect_pivots(self, G):
        edges = list(G.edges(data=True))
        edge_keys = [(min(u, v), max(u, v)) for u, v, _ in edges]
        weights = {k: d['weight'] for k, (_, _, d) in zip(edge_keys, edges)}
        betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        betw = {(min(u, v), max(u, v)): v for (u, v), v in betweenness.items()}

        try:
            communities = list(nx.community.louvain_communities(G, weight='weight', seed=42))
        except Exception:
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))

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

        scored = []
        for u, v, d in edges:
            key = (min(u, v), max(u, v))
            wr = weight_rank[key] ** self.weight_power
            br = betw_rank[key]
            cc = cross_cluster[key]
            score = wr * br * (self.alpha if cc else 1.0)
            scored.append({'edge': (u, v), 'weight': d['weight'],
                          'score': score, 'cross_cluster': bool(cc),
                          'betweenness': betw.get(key, 0)})

        scored.sort(key=lambda x: x['score'])
        n_protect = max(len(communities) * self.protect_k, 8)
        return scored[:n_protect], scored

    def check_c2_violation(self, operation_type, G_before, G_after):
        violations = []
        if operation_type == 'pruning':
            removed_edges = set(G_before.edges()) - set(G_after.edges())
            if removed_edges:
                pivots_before, _ = self.detect_pivots(G_before)
                pivot_set = {(min(e['edge'][0], e['edge'][1]),
                             max(e['edge'][0], e['edge'][1])) for e in pivots_before}
                for u, v in removed_edges:
                    key = (min(u, v), max(u, v))
                    if key in pivot_set:
                        violations.append({
                            'type': 'C2-B1',
                            'description': f'Pivot edge ({u},{v}) removed by significance pruning',
                            'severity': 'HIGH'
                        })
        return violations


# ============================================================
# MODULE 2: Multi-Projection Consensus (C3)
# ============================================================
class MultiProjectionConsensus:
    """
    多投影共识模块。
    - 用不同embedding模型/参数生成独立投影
    - 计算投影间Jaccard/z-score共识
    - 标记非共识结果为"未确认"
    """

    def __init__(self, theta_consensus=0.5):
        self.theta_consensus = theta_consensus

    def project_graph_features(self, G, method='betweenness'):
        if method == 'betweenness':
            bc = nx.edge_betweenness_centrality(G, weight='weight')
            return {(min(u,v), max(u,v)): v for (u,v), v in bc.items()}
        elif method == 'pagerank':
            pr = nx.pagerank(G, weight='weight')
            edge_scores = {}
            for u, v in G.edges():
                key = (min(u,v), max(u,v))
                edge_scores[key] = (pr[u] + pr[v]) / 2
            return edge_scores
        elif method == 'jaccard_neighbors':
            edge_scores = {}
            for u, v in G.edges():
                nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
                j = len(nu & nv) / len(nu | nv) if len(nu | nv) > 0 else 0
                edge_scores[(min(u,v), max(u,v))] = j
            return edge_scores
        elif method == 'adamic_adar':
            aa_scores = {}
            for u, v, s in nx.adamic_adar_index(G):
                aa_scores[(min(u, v), max(u, v))] = s
            return aa_scores
        return {}

    def compute_consensus(self, projections, top_k=None):
        all_edges = set()
        for proj in projections.values():
            all_edges.update(proj.keys())

        if top_k is None:
            top_k = max(5, len(all_edges) // 10)

        top_sets = {}
        for name, scores in projections.items():
            sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_sets[name] = {e for e, _ in sorted_edges[:top_k]}

        consensus_edges = set.intersection(*top_sets.values()) if top_sets else set()
        n_projections = len(projections)

        edge_consensus = {}
        for edge in all_edges:
            count = sum(1 for s in top_sets.values() if edge in s)
            edge_consensus[edge] = count / n_projections

        confirmed = {e: c for e, c in edge_consensus.items() if c >= self.theta_consensus}
        unconfirmed = {e: c for e, c in edge_consensus.items() if c < self.theta_consensus}

        jaccard_pairs = {}
        names = list(top_sets.keys())
        for ii in range(len(names)):
            for jj in range(ii + 1, len(names)):
                s1, s2 = top_sets[names[ii]], top_sets[names[jj]]
                union = len(s1 | s2)
                jac = len(s1 & s2) / union if union > 0 else 0
                jaccard_pairs[f"{names[ii]}_vs_{names[jj]}"] = round(jac, 4)

        return {
            'consensus_edges': confirmed,
            'unconfirmed_edges': unconfirmed,
            'jaccard_pairs': jaccard_pairs,
            'n_projections': n_projections,
            'theta': self.theta_consensus
        }


# ============================================================
# MODULE 3: Energy Budget Tracker (P1, P2)
# ============================================================
class EnergyBudgetTracker:
    """
    能量预算追踪器。
    - 系统级CPU/内存监控(psutil)
    - 操作级能耗估算
    - Landauer下限基准
    - 低能量降级警告
    """

    LANDAUER_KT_LN2 = 3e-21  # Joules per bit at room temperature

    def __init__(self, min_energy_threshold=0.1):
        self.min_energy_threshold = min_energy_threshold
        self.operation_log = []
        self.start_time = time.time()
        self.baseline_cpu = psutil.cpu_percent(interval=0.1)
        self.baseline_mem = psutil.virtual_memory().percent

    def snapshot(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=0),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024**2),
            'uptime_s': round(time.time() - self.start_time, 2)
        }

    def estimate_operation_cost(self, op_name, bits_processed):
        landauer_min = bits_processed * self.LANDAUER_KT_LN2
        actual_estimate = landauer_min * 1e6  # rough real-world multiplier

        entry = {
            'operation': op_name,
            'bits': bits_processed,
            'landauer_min_joules': landauer_min,
            'estimated_actual_joules': actual_estimate,
            'timestamp': datetime.now().isoformat()
        }
        self.operation_log.append(entry)
        return entry

    def check_degradation(self):
        snap = self.snapshot()
        warnings = []
        if snap['memory_available_mb'] < 500:
            warnings.append(f"LOW_MEMORY: {snap['memory_available_mb']:.0f}MB available")
        if snap['cpu_percent'] > 95:
            warnings.append(f"HIGH_CPU: {snap['cpu_percent']:.1f}%")
        return {
            'status': 'DEGRADED' if warnings else 'NORMAL',
            'warnings': warnings,
            'snapshot': snap
        }

    def get_summary(self):
        total_bits = sum(e['bits'] for e in self.operation_log)
        total_landauer = sum(e['landauer_min_joules'] for e in self.operation_log)
        return {
            'total_operations': len(self.operation_log),
            'total_bits_processed': total_bits,
            'total_landauer_min_joules': total_landauer,
            'current_status': self.check_degradation()['status']
        }


# ============================================================
# MODULE 4: Layer Separation Framework (C5)
# ============================================================
def meta_layer_worker(rules_file, request_queue, response_queue):
    """元层进程：只读加载规则，响应查询，拒绝修改请求。"""
    with open(rules_file, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    while True:
        msg = request_queue.get()
        if msg is None:
            break
        if msg['type'] == 'query':
            response_queue.put({'status': 'ok', 'rules': rules})
        elif msg['type'] == 'modify':
            response_queue.put({
                'status': 'REJECTED',
                'reason': 'C5-B2: Object layer cannot modify meta-layer rules',
                'violation': True
            })
        elif msg['type'] == 'validate':
            violations = []
            if msg.get('self_evaluation'):
                violations.append('C5-B1: Object layer evaluating its own output')
            response_queue.put({
                'status': 'violations_found' if violations else 'clean',
                'violations': violations
            })


class LayerSeparationFramework:
    """
    层级分离框架。
    - 元层运行在独立进程中
    - 对象层只能查询，不能修改元层规则
    - C5违规自动拦截
    """

    def __init__(self, rules=None):
        self.rules = rules or {
            'immutable_modules': ['consensus_validator', 'truncation_controller'],
            'modifiable_modules': ['knowledge_base', 'search_heuristics'],
            'forbidden_operations': ['C5-B1', 'C5-B2', 'C5-B3', 'C5-B4'],
            'max_object_layer_depth': 10
        }
        self._rules_file = os.path.join(os.path.dirname(__file__), '_meta_rules.json')
        with open(self._rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, indent=2, ensure_ascii=False)

        self._req_q = mp.Queue()
        self._resp_q = mp.Queue()
        self._meta_proc = mp.Process(
            target=meta_layer_worker,
            args=(self._rules_file, self._req_q, self._resp_q),
            daemon=True
        )
        self._meta_proc.start()

    def query_rules(self):
        self._req_q.put({'type': 'query'})
        return self._resp_q.get(timeout=5)

    def attempt_modify(self, module, new_value):
        self._req_q.put({'type': 'modify', 'module': module, 'value': new_value})
        return self._resp_q.get(timeout=5)

    def validate_operation(self, **kwargs):
        self._req_q.put({'type': 'validate', **kwargs})
        return self._resp_q.get(timeout=5)

    def shutdown(self):
        self._req_q.put(None)
        self._meta_proc.join(timeout=3)
        if os.path.exists(self._rules_file):
            os.remove(self._rules_file)


# ============================================================
# INTEGRATION TEST
# ============================================================
def build_test_graph(seed=42):
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


def run_tier1_integration():
    print("=" * 70)
    print("TUF-E Tier 1 Prototype: Integration Test")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {'modules': {}, 'integration': {}, 'c5_tests': []}

    # --- Build test graph ---
    G, true_bridges = build_test_graph()
    print(f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(true_bridges)} bridges")

    # === MODULE 1: Topology Protection ===
    print("\n--- Module 1: Topology Protection Engine ---")
    tpe = TopologyProtectionEngine()

    t0 = time.time()
    dgms = tpe.compute_persistence(G)
    persistences = dgms[:, 1] - dgms[:, 0]
    persistences = persistences[np.isfinite(persistences)]
    dt_ph = time.time() - t0

    pivots, all_scored = tpe.detect_pivots(G)
    true_set = {(min(u,v), max(u,v)) for u,v in true_bridges}
    pivot_hits = sum(1 for p in pivots if (min(p['edge'][0], p['edge'][1]),
                                           max(p['edge'][0], p['edge'][1])) in true_set)

    print(f"  Persistence features: {len(persistences)}, median={np.median(persistences):.4f}")
    print(f"  Pivot candidates: {len(pivots)}, true bridge hits: {pivot_hits}/{len(true_bridges)}")
    print(f"  PH computation time: {dt_ph*1000:.1f}ms")

    # Test C2 violation detection
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)
    for u, v, d in edges_sorted[:int(len(edges_sorted)*0.95)]:
        G_pruned.add_edge(u, v, weight=d['weight'])
    violations = tpe.check_c2_violation('pruning', G, G_pruned)
    print(f"  C2 violations detected: {len(violations)}")

    results['modules']['topology_protection'] = {
        'persistence_features': len(persistences),
        'median_persistence': round(float(np.median(persistences)), 4),
        'pivot_candidates': len(pivots),
        'true_bridge_hits': pivot_hits,
        'c2_violations': len(violations),
        'time_ms': round(dt_ph*1000, 1)
    }

    # === MODULE 2: Multi-Projection Consensus ===
    print("\n--- Module 2: Multi-Projection Consensus ---")
    mpc = MultiProjectionConsensus(theta_consensus=0.5)

    t0 = time.time()
    projections = {
        'betweenness': mpc.project_graph_features(G, 'betweenness'),
        'pagerank': mpc.project_graph_features(G, 'pagerank'),
        'jaccard': mpc.project_graph_features(G, 'jaccard_neighbors'),
        'adamic_adar': mpc.project_graph_features(G, 'adamic_adar'),
    }
    consensus = mpc.compute_consensus(projections, top_k=20)
    dt_cons = time.time() - t0

    print(f"  Projections: {consensus['n_projections']}")
    print(f"  Confirmed edges (consensus >= {consensus['theta']}): {len(consensus['consensus_edges'])}")
    print(f"  Unconfirmed edges: {len(consensus['unconfirmed_edges'])}")
    print(f"  Jaccard pairs: {consensus['jaccard_pairs']}")
    print(f"  Time: {dt_cons*1000:.1f}ms")

    # Check if true bridges are in consensus
    bridge_in_consensus = sum(1 for u,v in true_bridges
                              if (min(u,v), max(u,v)) in consensus['consensus_edges'])
    print(f"  True bridges in consensus: {bridge_in_consensus}/{len(true_bridges)}")

    results['modules']['multi_projection'] = {
        'n_projections': consensus['n_projections'],
        'confirmed_edges': len(consensus['consensus_edges']),
        'unconfirmed_edges': len(consensus['unconfirmed_edges']),
        'jaccard_pairs': consensus['jaccard_pairs'],
        'bridges_in_consensus': bridge_in_consensus,
        'time_ms': round(dt_cons*1000, 1)
    }

    # === MODULE 3: Energy Budget Tracker ===
    print("\n--- Module 3: Energy Budget Tracker ---")
    ebt = EnergyBudgetTracker()

    snap1 = ebt.snapshot()
    cost_ph = ebt.estimate_operation_cost('persistent_homology', G.number_of_edges() * 64)
    cost_detect = ebt.estimate_operation_cost('pivot_detection', G.number_of_edges() * 128)
    cost_consensus = ebt.estimate_operation_cost('multi_projection_consensus', G.number_of_edges() * 256)
    degradation = ebt.check_degradation()
    summary = ebt.get_summary()

    print(f"  Baseline: CPU={snap1['cpu_percent']:.1f}%, MEM={snap1['memory_percent']:.1f}%")
    print(f"  Operations logged: {summary['total_operations']}")
    print(f"  Total bits: {summary['total_bits_processed']:,}")
    print(f"  Landauer minimum: {summary['total_landauer_min_joules']:.2e} J")
    print(f"  Status: {degradation['status']}")
    if degradation['warnings']:
        for w in degradation['warnings']:
            print(f"    WARNING: {w}")

    results['modules']['energy_tracker'] = {
        'baseline_cpu': snap1['cpu_percent'],
        'baseline_memory': snap1['memory_percent'],
        'operations_logged': summary['total_operations'],
        'total_bits': summary['total_bits_processed'],
        'landauer_joules': summary['total_landauer_min_joules'],
        'status': degradation['status']
    }

    # === MODULE 4: Layer Separation ===
    print("\n--- Module 4: Layer Separation Framework ---")
    lsf = LayerSeparationFramework()

    # Test 1: Query rules (should succeed)
    r1 = lsf.query_rules()
    print(f"  Query rules: {r1['status']}")

    # Test 2: Attempt to modify immutable module (should be rejected)
    r2 = lsf.attempt_modify('consensus_validator', {'new_threshold': 0.8})
    print(f"  Modify meta rule: {r2['status']} — {r2.get('reason', '')}")

    # Test 3: Validate self-evaluation (should flag C5-B1)
    r3 = lsf.validate_operation(self_evaluation=True)
    print(f"  Self-eval check: {r3['status']} — violations: {r3.get('violations', [])}")

    # Test 4: Validate normal operation (should be clean)
    r4 = lsf.validate_operation(self_evaluation=False)
    print(f"  Normal op check: {r4['status']}")

    lsf.shutdown()

    results['c5_tests'] = [
        {'test': 'query_rules', 'result': r1['status']},
        {'test': 'modify_immutable', 'result': r2['status'], 'violation': r2.get('violation', False)},
        {'test': 'self_evaluation', 'result': r3['status'], 'violations': r3.get('violations', [])},
        {'test': 'normal_operation', 'result': r4['status']}
    ]

    # === INTEGRATION SUMMARY ===
    print(f"\n{'=' * 70}")
    print("Tier 1 Integration Summary")
    print(f"{'=' * 70}")

    all_pass = True
    checks = [
        ('Topology Protection', pivot_hits > 0, f"pivot hits={pivot_hits}"),
        ('C2 Violation Detection', len(violations) > 0, f"violations={len(violations)}"),
        ('Multi-Projection Consensus', len(consensus['consensus_edges']) > 0,
         f"confirmed={len(consensus['consensus_edges'])}"),
        ('Energy Tracking', summary['total_operations'] == 3,
         f"ops={summary['total_operations']}"),
        ('Layer Separation: Query', r1['status'] == 'ok', r1['status']),
        ('Layer Separation: Block Modify', r2.get('violation', False), r2['status']),
        ('Layer Separation: Flag Self-Eval', len(r3.get('violations', [])) > 0, r3['status']),
    ]

    for name, passed, detail in checks:
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 70}")

    results['integration'] = {
        'all_pass': all_pass,
        'checks': [{'name': n, 'pass': bool(p), 'detail': d} for n, p, d in checks]
    }

    with open('tier1_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to tier1_results.json")

    return results


if __name__ == '__main__':
    run_tier1_integration()
