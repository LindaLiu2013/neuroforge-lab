"""
TUF-E Tier 2 Prototype: 四模块扩展验证

在 Tier 1 基础上增加:
  A. DissipativeKnowledgeStore - 耗散知识库 (知识=远离平衡态稳态)
  B. OnlineSelfModification - 在线自修改引擎 (元层监督下的运行时变更)
  C. CategoryTheoreticPivot - 范畴论枢机检测 (不可分解态射识别)
  D. WeakLinkConsensus - 弱连接共识增强 (捕获低权重枢机)

Tier 1 模块作为依赖导入，不重复实现。
"""

import numpy as np
import networkx as nx
from ripser import ripser
import json
import time
import os
import sys
import io
import copy
import hashlib
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# MODULE A: Dissipative Knowledge Store (C1, P1, P2)
# ============================================================
class DissipativeKnowledgeStore:
    """
    耗散知识库。

    核心原理: 知识不是静态存储，而是远离平衡态的稳态模式。
    每条知识需要持续的能量投入来维持；停止维护 → 自然衰减 → 遗忘。
    遗忘是热力学必然，不是缺陷。

    与 Tier 1 EnergyBudgetTracker 集成:
    - 每次访问/强化消耗能量预算
    - 能量不足时加速衰减 (graceful degradation)
    - Landauer 下限作为理论基准
    """

    def __init__(self, decay_rate=0.05, reinforcement_boost=0.3,
                 min_viability=0.1, energy_per_access=64):
        self.decay_rate = decay_rate
        self.reinforcement_boost = reinforcement_boost
        self.min_viability = min_viability
        self.energy_per_access = energy_per_access  # bits per access
        self.store = {}
        self.access_log = []
        self.cycle_count = 0

    def ingest(self, key, content, initial_strength=1.0, source='observation'):
        self.store[key] = {
            'content': content,
            'strength': min(initial_strength, 1.0),
            'created_at': self.cycle_count,
            'last_accessed': self.cycle_count,
            'access_count': 0,
            'source': source,
            'energy_invested': self.energy_per_access
        }

    def access(self, key, reinforce=True):
        if key not in self.store:
            return None
        entry = self.store[key]
        entry['last_accessed'] = self.cycle_count
        entry['access_count'] += 1
        entry['energy_invested'] += self.energy_per_access
        if reinforce:
            entry['strength'] = min(1.0, entry['strength'] + self.reinforcement_boost)
        self.access_log.append({
            'key': key, 'cycle': self.cycle_count,
            'reinforced': reinforce, 'strength_after': entry['strength']
        })
        return entry

    def decay_step(self, energy_available=True):
        self.cycle_count += 1
        expired = []
        effective_decay = self.decay_rate if energy_available else self.decay_rate * 3
        for key, entry in list(self.store.items()):
            idle_cycles = self.cycle_count - entry['last_accessed']
            decay = effective_decay * (1 + 0.01 * idle_cycles)
            entry['strength'] -= decay
            if entry['strength'] < self.min_viability:
                expired.append(key)
        for key in expired:
            del self.store[key]
        return {
            'cycle': self.cycle_count,
            'active_entries': len(self.store),
            'expired': len(expired),
            'expired_keys': expired,
            'effective_decay': effective_decay
        }

    def get_total_energy_invested(self):
        return sum(e['energy_invested'] for e in self.store.values())

    def get_summary(self):
        if not self.store:
            return {'entries': 0, 'mean_strength': 0, 'total_energy': 0}
        strengths = [e['strength'] for e in self.store.values()]
        return {
            'entries': len(self.store),
            'mean_strength': round(float(np.mean(strengths)), 4),
            'min_strength': round(float(np.min(strengths)), 4),
            'max_strength': round(float(np.max(strengths)), 4),
            'total_energy_bits': self.get_total_energy_invested(),
            'cycles_elapsed': self.cycle_count,
            'total_accesses': len(self.access_log)
        }

    def stress_test(self, n_entries=50, n_cycles=100, access_rate=0.3,
                    energy_cutoff_cycle=None):
        rng = np.random.default_rng(42)
        for i in range(n_entries):
            self.ingest(f"fact_{i:03d}", f"content_{i}", initial_strength=rng.uniform(0.5, 1.0))

        trajectory = []
        for cycle in range(n_cycles):
            energy_ok = True
            if energy_cutoff_cycle and cycle >= energy_cutoff_cycle:
                energy_ok = False
            for key in list(self.store.keys()):
                if rng.random() < access_rate:
                    self.access(key, reinforce=True)
            result = self.decay_step(energy_available=energy_ok)
            trajectory.append({
                'cycle': cycle,
                'active': result['active_entries'],
                'expired': result['expired'],
                'energy_ok': energy_ok
            })
        return trajectory


# ============================================================
# MODULE B: Online Self-Modification Engine (C5)
# ============================================================
class OnlineSelfModification:
    """
    在线自修改引擎。

    对象层可以提议对自身参数的修改，但必须经过元层验证。
    元层检查:
      1. 目标参数是否在可修改列表中
      2. 新值是否在允许范围内
      3. 修改是否会导致性能退化

    三层状态: PROPOSED → VALIDATED → APPLIED / REJECTED / ROLLED_BACK
    """

    MODIFIABLE_PARAMS = {
        'decay_rate': {'min': 0.001, 'max': 0.5, 'default': 0.05},
        'reinforcement_boost': {'min': 0.01, 'max': 1.0, 'default': 0.3},
        'protect_k': {'min': 1, 'max': 50, 'default': 5},
        'alpha': {'min': 0.01, 'max': 1.0, 'default': 0.2},
        'weight_power': {'min': 0.5, 'max': 5.0, 'default': 2.0},
        'theta_consensus': {'min': 0.1, 'max': 1.0, 'default': 0.5},
    }

    IMMUTABLE_PARAMS = {
        'min_viability', 'energy_per_access', 'LANDAUER_KT_LN2',
        'immutable_modules', 'forbidden_operations'
    }

    def __init__(self):
        self.current_params = {k: v['default'] for k, v in self.MODIFIABLE_PARAMS.items()}
        self.modification_log = []
        self.param_history = [copy.deepcopy(self.current_params)]

    def propose_modification(self, param_name, new_value, reason='auto-tune'):
        proposal = {
            'param': param_name,
            'old_value': self.current_params.get(param_name),
            'new_value': new_value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'status': 'PROPOSED'
        }

        if param_name in self.IMMUTABLE_PARAMS:
            proposal['status'] = 'REJECTED'
            proposal['rejection_reason'] = f'C5-B2: {param_name} is immutable'
            self.modification_log.append(proposal)
            return proposal

        if param_name not in self.MODIFIABLE_PARAMS:
            proposal['status'] = 'REJECTED'
            proposal['rejection_reason'] = f'Unknown parameter: {param_name}'
            self.modification_log.append(proposal)
            return proposal

        spec = self.MODIFIABLE_PARAMS[param_name]
        if not (spec['min'] <= new_value <= spec['max']):
            proposal['status'] = 'REJECTED'
            proposal['rejection_reason'] = f'Value {new_value} out of range [{spec["min"]}, {spec["max"]}]'
            self.modification_log.append(proposal)
            return proposal

        proposal['status'] = 'VALIDATED'
        self.modification_log.append(proposal)
        return proposal

    def apply_modification(self, proposal):
        if proposal['status'] != 'VALIDATED':
            return {'status': 'ERROR', 'reason': 'Proposal not validated'}

        param = proposal['param']
        old_val = self.current_params[param]
        self.current_params[param] = proposal['new_value']
        self.param_history.append(copy.deepcopy(self.current_params))

        proposal['status'] = 'APPLIED'
        return {
            'status': 'APPLIED',
            'param': param,
            'old': old_val,
            'new': proposal['new_value']
        }

    def rollback(self, steps=1):
        if len(self.param_history) <= 1:
            return {'status': 'ERROR', 'reason': 'Nothing to rollback'}
        for _ in range(min(steps, len(self.param_history) - 1)):
            self.param_history.pop()
        self.current_params = copy.deepcopy(self.param_history[-1])
        return {
            'status': 'ROLLED_BACK',
            'current_params': copy.deepcopy(self.current_params),
            'history_depth': len(self.param_history)
        }

    def validate_performance(self, metric_before, metric_after, threshold=0.8):
        ratio = metric_after / metric_before if metric_before > 0 else 0
        if ratio < threshold:
            rb = self.rollback()
            return {
                'status': 'DEGRADED_ROLLED_BACK',
                'ratio': round(ratio, 4),
                'threshold': threshold,
                'rollback': rb
            }
        return {
            'status': 'ACCEPTED',
            'ratio': round(ratio, 4),
            'threshold': threshold
        }

    def get_summary(self):
        statuses = defaultdict(int)
        for m in self.modification_log:
            statuses[m['status']] += 1
        return {
            'total_proposals': len(self.modification_log),
            'by_status': dict(statuses),
            'current_params': copy.deepcopy(self.current_params),
            'history_depth': len(self.param_history)
        }


# ============================================================
# MODULE C: Category-Theoretic Pivot Detection (Q0.4)
# ============================================================
class CategoryTheoreticPivot:
    """
    范畴论枢机检测器。

    将图建模为范畴:
      - 对象 = 节点
      - 态射 = 边 (带权重的有向/无向连接)
      - 复合 = 路径 (权重组合)

    不可分解态射 (irreducible morphism):
      一条边 e: A→B 是不可分解的，如果不存在中间对象 C
      使得 e = f∘g 其中 g:A→C, f:C→B 且 f,g 都不是恒等态射。

    在加权图中，这等价于: 边 (u,v) 是最短路径的唯一实现，
    即不存在替代路径 u→...→v 其组合权重 ≥ w(u,v)。

    这比启发式融合更有原则性: 它直接测试"这条边是否可以被绕过"。
    """

    def __init__(self, composition_rule='min', irreducibility_threshold=0.9):
        self.composition_rule = composition_rule
        self.irreducibility_threshold = irreducibility_threshold

    def compute_irreducibility(self, G):
        edge_scores = {}
        edges = list(G.edges(data=True))

        for u, v, d in edges:
            w_direct = d['weight']
            key = (min(u, v), max(u, v))

            G_temp = G.copy()
            G_temp.remove_edge(u, v)

            try:
                if self.composition_rule == 'min':
                    alt_path_len = nx.shortest_path_length(G_temp, u, v, weight='weight')
                elif self.composition_rule == 'max':
                    lengths = list(nx.all_simple_paths(G_temp, u, v, cutoff=5))
                    if lengths:
                        alt_path_len = max(
                            sum(G_temp[a][b]['weight'] for a, b in zip(p[:-1], p[1:]))
                            for p in lengths[:20]
                        )
                    else:
                        alt_path_len = float('inf')
                else:
                    alt_path_len = nx.shortest_path_length(G_temp, u, v, weight='weight')
            except nx.NetworkXNoPath:
                alt_path_len = float('inf')

            if alt_path_len == float('inf'):
                irreducibility = 1.0
            else:
                if self.composition_rule == 'min':
                    irreducibility = 1.0 - (w_direct / alt_path_len) if alt_path_len > 0 else 0.0
                else:
                    irreducibility = 1.0 - (alt_path_len / w_direct) if w_direct > 0 else 0.0
                irreducibility = max(0.0, min(1.0, irreducibility))

            edge_scores[key] = {
                'irreducibility': irreducibility,
                'direct_weight': w_direct,
                'alt_path_weight': alt_path_len if alt_path_len != float('inf') else None,
                'is_bridge': alt_path_len == float('inf')
            }

        return edge_scores

    def detect_pivots(self, G, top_k=None):
        scores = self.compute_irreducibility(G)
        ranked = sorted(scores.items(), key=lambda x: x[1]['irreducibility'], reverse=True)

        if top_k is None:
            top_k = max(5, len(ranked) // 10)

        pivots = []
        for edge_key, data in ranked[:top_k]:
            if data['irreducibility'] >= self.irreducibility_threshold or data['is_bridge']:
                pivots.append((edge_key, data))

        return pivots, scores

    def compare_with_heuristic(self, G, heuristic_pivots, true_bridges):
        cat_pivots, cat_scores = self.detect_pivots(G)
        cat_pivot_set = {k for k, _ in cat_pivots}
        true_set = {(min(u, v), max(u, v)) for u, v in true_bridges}
        heur_set = {(min(e['edge'][0], e['edge'][1]),
                     max(e['edge'][0], e['edge'][1])) for e in heuristic_pivots}

        cat_hits = len(cat_pivot_set & true_set)
        heur_hits = len(heur_set & true_set)

        cat_only = cat_pivot_set - heur_set
        heur_only = heur_set - cat_pivot_set
        both = cat_pivot_set & heur_set

        return {
            'category_hits': cat_hits,
            'heuristic_hits': heur_hits,
            'true_bridges': len(true_bridges),
            'overlap': len(both),
            'cat_only': len(cat_only),
            'heur_only': len(heur_only),
            'cat_efficiency': cat_hits / heur_hits if heur_hits > 0 else 0,
            'cat_pivot_count': len(cat_pivots),
            'heur_pivot_count': len(heuristic_pivots)
        }


# ============================================================
# MODULE D: Weak-Link Consensus Enhancement (C3 fix)
# ============================================================
class WeakLinkConsensus:
    """
    弱连接共识增强。

    Tier 1 的问题: 标准投影方法(betweenness/pagerank/jaccard/adamic_adar)
    都偏向高权重边，导致低权重桥接边在共识中被淹没。

    解决方案: 增加对弱连接敏感的投影方法:
      1. inverse_weight: 低权重 = 高分 (直接反转)
      2. community_bridge: 跨社区边获得额外加分
      3. persistence_weighted: PH persistence 低的边更可能是枢机
      4. structural_hole: Burt 结构洞指标 — 连接不同簇的边

    共识投票改为加权: 弱连接敏感方法获得更高投票权重。
    """

    def __init__(self, weak_link_weight=2.0, standard_weight=1.0, theta=0.4):
        self.weak_link_weight = weak_link_weight
        self.standard_weight = standard_weight
        self.theta = theta

    def project_inverse_weight(self, G):
        scores = {}
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        max_w = max(weights) if weights else 1.0
        for u, v, d in G.edges(data=True):
            key = (min(u, v), max(u, v))
            scores[key] = max_w - d['weight']
        return scores

    def project_community_bridge(self, G):
        try:
            communities = list(nx.community.louvain_communities(G, weight='weight', seed=42))
        except Exception:
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))

        node_comm = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = cid

        scores = {}
        for u, v, d in G.edges(data=True):
            key = (min(u, v), max(u, v))
            is_cross = node_comm.get(u, -1) != node_comm.get(v, -1)
            scores[key] = 1.0 if is_cross else 0.0
        return scores

    def project_structural_hole(self, G):
        scores = {}
        for u, v, d in G.edges(data=True):
            key = (min(u, v), max(u, v))
            nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
            shared = len(nu & nv)
            total = len(nu | nv)
            constraint = shared / total if total > 0 else 0
            scores[key] = 1.0 - constraint
        return scores

    def project_persistence_aware(self, G):
        distances = nx.to_numpy_array(G, weight='weight')
        np.fill_diagonal(distances, 0)
        max_w = distances.max() if distances.max() > 0 else 1.0
        dist_matrix = max_w - distances
        np.fill_diagonal(dist_matrix, 0)

        try:
            result = ripser(dist_matrix, maxdim=0)
            dgms = result['dgms'][0]
            persistences = dgms[:, 1] - dgms[:, 0]
            persistences = persistences[np.isfinite(persistences)]
            median_p = np.median(persistences) if len(persistences) > 0 else 1.0
        except Exception:
            median_p = 1.0

        scores = {}
        for u, v, d in G.edges(data=True):
            key = (min(u, v), max(u, v))
            low_persistence_score = 1.0 / (1.0 + d['weight'] / median_p)
            scores[key] = low_persistence_score
        return scores

    def compute_enhanced_consensus(self, G, top_k=None):
        projections = {
            'inverse_weight': (self.project_inverse_weight(G), self.weak_link_weight),
            'community_bridge': (self.project_community_bridge(G), self.weak_link_weight),
            'structural_hole': (self.project_structural_hole(G), self.weak_link_weight),
            'persistence_aware': (self.project_persistence_aware(G), self.weak_link_weight),
            'betweenness': (
                {(min(u,v), max(u,v)): v for (u,v), v in nx.edge_betweenness_centrality(G, weight='weight').items()},
                self.standard_weight
            ),
            'pagerank': (
                {(min(u,v), max(u,v)): (nx.pagerank(G, weight='weight')[u] + nx.pagerank(G, weight='weight')[v]) / 2
                 for u, v in G.edges()},
                self.standard_weight
            ),
        }

        all_edges = set()
        for proj, _ in projections.values():
            all_edges.update(proj.keys())

        if top_k is None:
            top_k = max(5, len(all_edges) // 10)

        weighted_votes = defaultdict(float)
        total_weight = 0
        for name, (scores, weight) in projections.items():
            sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_edges = {e for e, _ in sorted_edges[:top_k]}
            for edge in top_edges:
                weighted_votes[edge] += weight
            total_weight += weight

        edge_consensus = {e: v / total_weight for e, v in weighted_votes.items()}
        confirmed = {e: c for e, c in edge_consensus.items() if c >= self.theta}
        unconfirmed = {e: c for e, c in edge_consensus.items() if c < self.theta}

        return {
            'consensus_edges': confirmed,
            'unconfirmed_edges': unconfirmed,
            'n_projections': len(projections),
            'projection_weights': {name: w for name, (_, w) in projections.items()},
            'theta': self.theta
        }


# ============================================================
# TIER 1 IMPORTS (inline to avoid import issues)
# ============================================================
class TopologyProtectionEngine:
    def __init__(self, alpha=0.2, weight_power=2.0, protect_k=5):
        self.alpha = alpha
        self.weight_power = weight_power
        self.protect_k = protect_k

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


def build_test_graph(seed=42):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    true_bridges = []
    n_per_basin, n_basins, n_bridge_edges = 30, 2, 5
    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j_idx in range(i + 1, n_per_basin):
                if rng.random() < 0.4:
                    G.add_edge(offset + i, offset + j_idx, weight=rng.uniform(0.7, 1.0))
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


# ============================================================
# INTEGRATION TEST
# ============================================================
def run_tier2_integration():
    print("=" * 70)
    print("TUF-E Tier 2 Prototype: Extended Integration Test")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {'modules': {}, 'integration': {}, 'comparison': {}}
    G, true_bridges = build_test_graph()
    true_set = {(min(u, v), max(u, v)) for u, v in true_bridges}
    print(f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(true_bridges)} bridges")

    # === MODULE A: Dissipative Knowledge Store ===
    print("\n--- Module A: Dissipative Knowledge Store ---")
    t0 = time.time()
    dks = DissipativeKnowledgeStore(decay_rate=0.05, reinforcement_boost=0.3)

    trajectory = dks.stress_test(n_entries=50, n_cycles=100, access_rate=0.3, energy_cutoff_cycle=70)
    dt_dks = time.time() - t0

    final_summary = dks.get_summary()
    entries_at_cutoff = next((t['active'] for t in trajectory if t['cycle'] == 70), None)
    entries_final = trajectory[-1]['active']
    entries_no_energy = entries_final

    steady_state_active = any(t['active'] > 0 and t['active'] < 50 for t in trajectory[10:70])
    forgetting_on_cutoff = entries_final < (entries_at_cutoff or 50)

    print(f"  Entries ingested: 50")
    print(f"  At energy cutoff (cycle 70): {entries_at_cutoff} active")
    print(f"  Final (cycle 100, no energy): {entries_final} active")
    print(f"  Steady state achieved: {steady_state_active}")
    print(f"  Forgetting on energy cutoff: {forgetting_on_cutoff}")
    print(f"  Mean strength: {final_summary['mean_strength']}")
    print(f"  Total energy invested: {final_summary.get('total_energy_bits', dks.get_total_energy_invested())} bits")
    print(f"  Time: {dt_dks*1000:.1f}ms")

    results['modules']['dissipative_knowledge'] = {
        'entries_ingested': 50,
        'at_cutoff': entries_at_cutoff,
        'final_active': entries_final,
        'steady_state': steady_state_active,
        'forgetting_works': forgetting_on_cutoff,
        'mean_strength': final_summary['mean_strength'],
        'total_energy_bits': dks.get_total_energy_invested(),
        'time_ms': round(dt_dks * 1000, 1)
    }

    # === MODULE B: Online Self-Modification ===
    print("\n--- Module B: Online Self-Modification Engine ---")
    t0 = time.time()
    osm = OnlineSelfModification()

    p1 = osm.propose_modification('decay_rate', 0.1, reason='slow decay test')
    r1 = osm.apply_modification(p1)

    p2 = osm.propose_modification('min_viability', 0.5, reason='should be rejected')

    p3 = osm.propose_modification('alpha', 2.0, reason='out of range test')

    p4 = osm.propose_modification('protect_k', 10, reason='valid change')
    r4 = osm.apply_modification(p4)

    perf_result = osm.validate_performance(metric_before=1.0, metric_after=0.5, threshold=0.8)

    summary_osm = osm.get_summary()
    dt_osm = time.time() - t0

    valid_applied = r1['status'] == 'APPLIED'
    immutable_rejected = p2['status'] == 'REJECTED'
    range_rejected = p3['status'] == 'REJECTED'
    degradation_rolled_back = perf_result['status'] == 'DEGRADED_ROLLED_BACK'

    print(f"  Valid modification: {r1['status']}")
    print(f"  Immutable rejection: {p2['status']} — {p2.get('rejection_reason', '')}")
    print(f"  Range rejection: {p3['status']} — {p3.get('rejection_reason', '')}")
    print(f"  Degradation rollback: {perf_result['status']} (ratio={perf_result['ratio']})")
    print(f"  Total proposals: {summary_osm['total_proposals']}")
    print(f"  Status breakdown: {summary_osm['by_status']}")
    print(f"  Time: {dt_osm*1000:.1f}ms")

    results['modules']['self_modification'] = {
        'valid_applied': valid_applied,
        'immutable_rejected': immutable_rejected,
        'range_rejected': range_rejected,
        'degradation_rolled_back': degradation_rolled_back,
        'total_proposals': summary_osm['total_proposals'],
        'by_status': summary_osm['by_status'],
        'time_ms': round(dt_osm * 1000, 1)
    }

    # === MODULE C: Category-Theoretic Pivot Detection ===
    print("\n--- Module C: Category-Theoretic Pivot Detection ---")
    t0 = time.time()
    ctp = CategoryTheoreticPivot(composition_rule='min', irreducibility_threshold=0.5)

    tpe = TopologyProtectionEngine()
    heuristic_pivots, _ = tpe.detect_pivots(G)

    comparison = ctp.compare_with_heuristic(G, heuristic_pivots, true_bridges)
    dt_cat = time.time() - t0

    print(f"  Category-theoretic hits: {comparison['category_hits']}/{comparison['true_bridges']}")
    print(f"  Heuristic hits: {comparison['heuristic_hits']}/{comparison['true_bridges']}")
    print(f"  Overlap: {comparison['overlap']}")
    print(f"  Cat-only: {comparison['cat_only']}, Heur-only: {comparison['heur_only']}")
    print(f"  Efficiency (cat/heur): {comparison['cat_efficiency']:.1%}")
    print(f"  Cat pivots: {comparison['cat_pivot_count']}, Heur pivots: {comparison['heur_pivot_count']}")
    print(f"  Time: {dt_cat*1000:.1f}ms")

    results['modules']['category_pivot'] = comparison
    results['modules']['category_pivot']['time_ms'] = round(dt_cat * 1000, 1)
    results['comparison'] = comparison

    # === MODULE D: Weak-Link Consensus ===
    print("\n--- Module D: Weak-Link Consensus Enhancement ---")
    t0 = time.time()
    wlc = WeakLinkConsensus(weak_link_weight=2.0, standard_weight=1.0, theta=0.4)

    enhanced = wlc.compute_enhanced_consensus(G, top_k=20)
    bridge_in_enhanced = sum(1 for u, v in true_bridges
                             if (min(u, v), max(u, v)) in enhanced['consensus_edges'])
    dt_wlc = time.time() - t0

    print(f"  Projections: {enhanced['n_projections']} (4 weak-link + 2 standard)")
    print(f"  Confirmed edges: {len(enhanced['consensus_edges'])}")
    print(f"  Bridges in consensus: {bridge_in_enhanced}/{len(true_bridges)}")
    print(f"  Projection weights: {enhanced['projection_weights']}")
    print(f"  Theta: {enhanced['theta']}")
    print(f"  Time: {dt_wlc*1000:.1f}ms")

    results['modules']['weak_link_consensus'] = {
        'n_projections': enhanced['n_projections'],
        'confirmed_edges': len(enhanced['consensus_edges']),
        'bridges_in_consensus': bridge_in_enhanced,
        'true_bridges': len(true_bridges),
        'projection_weights': enhanced['projection_weights'],
        'theta': enhanced['theta'],
        'time_ms': round(dt_wlc * 1000, 1)
    }

    # === INTEGRATION SUMMARY ===
    print(f"\n{'=' * 70}")
    print("Tier 2 Integration Summary")
    print(f"{'=' * 70}")

    checks = [
        ('A: Steady State', steady_state_active, f"active entries maintained between cycles 10-70"),
        ('A: Forgetting', forgetting_on_cutoff, f"{entries_at_cutoff}→{entries_final} after energy cutoff"),
        ('B: Valid Apply', valid_applied, f"decay_rate modified successfully"),
        ('B: Immutable Block', immutable_rejected, f"min_viability modification rejected"),
        ('B: Range Check', range_rejected, f"alpha=2.0 out of range rejected"),
        ('B: Degradation Rollback', degradation_rolled_back, f"perf ratio={perf_result['ratio']}"),
        ('C: Cat Pivot Detection', comparison['category_hits'] > 0,
         f"{comparison['category_hits']}/{comparison['true_bridges']} hits"),
        ('D: Weak-Link Consensus', bridge_in_enhanced >= 1,
         f"{bridge_in_enhanced}/{len(true_bridges)} bridges captured"),
    ]

    all_pass = True
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

    with open('tier2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to tier2_results.json")

    return results


if __name__ == '__main__':
    run_tier2_integration()
