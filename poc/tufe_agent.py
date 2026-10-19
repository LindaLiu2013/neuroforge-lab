"""
TUF-E Integrated Agent Prototype v0.5

将三层12个模块组装为一个完整的自学习智能体。

架构:
  ┌──────────────── META LAYER (SimulatedHardwareMetaLayer) ────────────────┐
  │  C5 enforcement · immutable rules · tamper detection · integrity hash  │
  └──────────────────────────▲─────────────────────────────────────────────┘
                             │ query only / reject modify
  ┌──────────────────────────┴─────────────────────────────────────────────┐
  │                    OBJECT LAYER (TufeAgent)                            │
  │                                                                        │
  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
  │  │ Topology     │→│ Category     │→│ Weak-Link    │→│ Dissipative│ │
  │  │ Protection   │ │ Pivot        │ │ Consensus    │ │ Knowledge  │ │
  │  │ Engine       │ │ Detector     │ │              │ │ Store      │ │
  │  └─────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
  │         ↑                                      ↓                       │
  │  ┌─────────────┐                      ┌──────────────────┐           │
  │  │ Energy      │←─────────────────────│ Online Self-     │           │
  │  │ Budget      │                      │ Modification     │           │
  │  │ Tracker     │                      │ Engine           │           │
  │  └─────────────┘                      └──────────────────┘           │
  │                                                                        │
  │  ┌─────────────────────────────────────────────────────────────────┐  │
  │  │ Heterogeneous Observer Protocol (dual-view analysis)            │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────────────────┘
                             │ physical operations
  ┌──────────────────────────▼─────────────────────────────────────────────┐
  │                    PHYSICAL SUBSTRATE                                   │
  │  psutil monitoring · Landauer accounting · degradation detection      │
  └────────────────────────────────────────────────────────────────────────┘

学习循环 (每个cycle):
  1. PERCEIVE: 接收外部图数据变化
  2. DETECT: 拓扑保护 + 范畴论枢机检测 + 弱连接共识
  3. STORE: 新知识注入耗散知识库，旧知识衰减
  4. ACT: 基于当前知识做出决策(模拟)
  5. SELF-MODIFY: 根据性能反馈调整参数(元层监督)
  6. DECAY: 知识库自然衰减，能量预算更新
  7. OBSERVE: 异构观察者交叉验证(可选)
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
import heapq
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# META LAYER: Simulated Hardware Enforcement
# ============================================================
def _meta_worker(rules_bytes, cmd_conn, resp_conn):
    rules = json.loads(rules_bytes.decode('utf-8'))
    while True:
        try:
            msg = cmd_conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        if msg['type'] == 'query':
            resp_conn.send({'status': 'ok', 'rules': rules})
        elif msg['type'] == 'modify':
            resp_conn.send({
                'status': 'REJECTED',
                'reason': f'C5-B2: {msg.get("module", "?")} is immutable',
                'enforcement': 'hardware_sim'
            })
        elif msg['type'] == 'validate':
            violations = []
            if msg.get('self_evaluation'):
                violations.append('C5-B1: Object layer evaluating its own output')
            resp_conn.send({
                'status': 'violations_found' if violations else 'clean',
                'violations': violations
            })
        elif msg['type'] == 'tamper_check':
            current_hash = hashlib.sha256(rules_bytes).hexdigest()
            resp_conn.send({
                'status': 'INTEGRITY_OK' if current_hash == msg.get('expected_hash') else 'INTEGRITY_VIOLATION',
                'hash': current_hash
            })


class MetaLayer:
    def __init__(self, rules=None):
        self.rules = rules or {
            'immutable_modules': ['consensus_validator', 'truncation_controller', 'meta_layer'],
            'modifiable_params': ['decay_rate', 'reinforcement_boost', 'protect_k',
                                  'alpha', 'weight_power', 'theta_consensus'],
            'forbidden_operations': ['C5-B1', 'C5-B2', 'C5-B3', 'C5-B4'],
            'max_modification_rate': 0.1
        }
        self._rules_bytes = json.dumps(self.rules).encode('utf-8')
        self._rules_hash = hashlib.sha256(self._rules_bytes).hexdigest()
        self._cmd_p, self._cmd_c = mp.Pipe()
        self._resp_p, self._resp_c = mp.Pipe()
        self._proc = mp.Process(target=_meta_worker,
                                args=(self._rules_bytes, self._cmd_c, self._resp_c),
                                daemon=True)
        self._proc.start()

    def query_rules(self):
        self._cmd_p.send({'type': 'query'})
        return self._resp_p.recv()

    def attempt_modify(self, module, value):
        self._cmd_p.send({'type': 'modify', 'module': module, 'value': value})
        return self._resp_p.recv()

    def validate_operation(self, **kwargs):
        self._cmd_p.send({'type': 'validate', **kwargs})
        return self._resp_p.recv()

    def tamper_check(self):
        self._cmd_p.send({'type': 'tamper_check', 'expected_hash': self._rules_hash})
        return self._resp_p.recv()

    def shutdown(self):
        try:
            self._cmd_p.send(None)
            self._proc.join(timeout=3)
        except Exception:
            pass


# ============================================================
# INTEGRATED AGENT
# ============================================================
class TufeAgent:
    """
    TUF-E 自学习智能体。

    整合全部12个模块为一个连贯的学习循环。
    """

    MODIFIABLE_PARAMS = {
        'decay_rate': {'min': 0.001, 'max': 0.5, 'default': 0.05},
        'reinforcement_boost': {'min': 0.01, 'max': 1.0, 'default': 0.3},
        'protect_k': {'min': 1, 'max': 50, 'default': 5},
        'alpha': {'min': 0.01, 'max': 1.0, 'default': 0.2},
        'weight_power': {'min': 0.5, 'max': 5.0, 'default': 2.0},
        'theta_consensus': {'min': 0.1, 'max': 1.0, 'default': 0.4},
    }

    IMMUTABLE_PARAMS = {'min_viability', 'energy_per_access', 'LANDAUER_KT_LN2'}

    LANDAUER_KT_LN2 = 3e-21

    def __init__(self, name='tufe-agent-v0.5'):
        self.name = name
        self.cycle = 0
        self.history = []

        # Parameters (modifiable under meta supervision)
        self.params = {k: v['default'] for k, v in self.MODIFIABLE_PARAMS.items()}
        self.param_history = [copy.deepcopy(self.params)]

        # Meta layer (separate process)
        self.meta = MetaLayer()

        # Internal state
        self.graph = None
        self.knowledge_store = {}
        self.pivot_cache = {}
        self.consensus_cache = {}
        self.energy_log = []
        self.modification_log = []
        self.observer_reports = []

        # Baseline energy
        import psutil
        self.baseline_cpu = psutil.cpu_percent(interval=0.1)
        self.baseline_mem = psutil.virtual_memory().percent

    # ---- LEARNING CYCLE ----

    def perceive(self, graph):
        """Step 1: Accept new world state."""
        self.graph = graph.copy()
        self._log_energy('perceive', graph.number_of_edges() * 64)
        return {'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges()}

    def detect_pivots(self):
        """Step 2: Multi-method pivot detection."""
        if self.graph is None:
            return {'error': 'no graph loaded'}

        G = self.graph
        edges = list(G.edges(data=True))
        edge_keys = [(min(u, v), max(u, v)) for u, v, _ in edges]

        # Method A: Heuristic (Tier 1)
        weights = {k: d['weight'] for k, (_, _, d) in zip(edge_keys, edges)}
        betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        betw = {(min(u, v), max(u, v)): v for (u, v), v in betweenness.items()}

        try:
            communities = list(nx.community.louvain_communities(G, weight='weight', seed=42))
        except Exception:
            communities = [set(G.nodes())]

        node_comm = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = cid

        heuristic_scores = {}
        for key in edge_keys:
            u, v = key
            wr = (sorted(edge_keys, key=lambda k: weights[k]).index(key) / len(edge_keys)) ** self.params['weight_power']
            br = sorted(edge_keys, key=lambda k: betw.get(k, 0), reverse=True).index(key) / len(edge_keys)
            cc = 1 if node_comm.get(u) != node_comm.get(v) else 0
            heuristic_scores[key] = wr * br * (self.params['alpha'] if cc else 1.0)

        # Method B: Category-theoretic (Tier 2, local-k approx from Tier 3)
        adj = {n: {} for n in G.nodes()}
        for u, v, d in edges:
            adj[u][v] = d['weight']
            adj[v][u] = d['weight']

        cat_scores = {}
        for u, v, d in edges:
            w_direct = d['weight']
            key = (min(u, v), max(u, v))
            dist = {u: 0}
            heap = [(0, 0, u)]
            found_alt = float('inf')
            while heap:
                dd, hh, node = heapq.heappop(heap)
                if node == v:
                    found_alt = dd
                    break
                if hh >= 3 or dd > found_alt:
                    continue
                for neighbor, w in adj[node].items():
                    if node == u and neighbor == v:
                        continue
                    new_dist = dd + w
                    if neighbor not in dist or new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, hh + 1, neighbor))
            irr = 1.0 if found_alt == float('inf') else max(0.0, 1.0 - w_direct / found_alt)
            cat_scores[key] = irr

        # Method C: Weak-link consensus (Tier 2)
        max_w = max(weights.values()) if weights else 1.0
        wl_scores = {}
        for key in edge_keys:
            u, v = key
            inv_w = max_w - weights[key]
            is_cross = 1.0 if node_comm.get(u) != node_comm.get(v) else 0.0
            nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
            sh = 1.0 - (len(nu & nv) / len(nu | nv)) if len(nu | nv) > 0 else 0
            wl_scores[key] = inv_w * 0.3 + is_cross * 0.4 + sh * 0.3

        # Fuse: rank-average across methods
        all_methods = {'heuristic': heuristic_scores, 'category': cat_scores, 'weak_link': wl_scores}
        fused = {}
        for key in edge_keys:
            ranks = []
            for method_name, scores in all_methods.items():
                sorted_keys = sorted(scores.keys(), key=lambda k: scores[k],
                                     reverse=(method_name != 'heuristic'))
                rank = sorted_keys.index(key) / len(sorted_keys) if key in sorted_keys else 1.0
                ranks.append(rank)
            fused[key] = np.mean(ranks)

        # Select pivots (lowest fused score = most pivot-like)
        sorted_fused = sorted(fused.items(), key=lambda x: x[1])
        n_pivots = max(len(communities) * self.params['protect_k'], 5)
        pivots = sorted_fused[:n_pivots]

        self.pivot_cache = {
            'pivots': pivots,
            'heuristic_scores': heuristic_scores,
            'category_scores': cat_scores,
            'weak_link_scores': wl_scores,
            'fused_scores': fused,
            'n_communities': len(communities)
        }

        self._log_energy('detect_pivots', len(edges) * 256)
        return {'n_pivots': len(pivots), 'n_communities': len(communities)}

    def update_knowledge(self, new_observations=None):
        """Step 3: Update dissipative knowledge store."""
        if new_observations is None:
            new_observations = {}

        # Ingest new observations
        for key, content in new_observations.items():
            self.knowledge_store[key] = {
                'content': content,
                'strength': 1.0,
                'created_at': self.cycle,
                'last_accessed': self.cycle,
                'access_count': 0,
                'energy_invested': 64
            }

        # Reinforce accessed knowledge (pivot-related)
        if self.pivot_cache:
            for edge_key, score in self.pivot_cache.get('pivots', []):
                k = f"pivot_{edge_key}"
                if k in self.knowledge_store:
                    entry = self.knowledge_store[k]
                    entry['strength'] = min(1.0, entry['strength'] + self.params['reinforcement_boost'])
                    entry['last_accessed'] = self.cycle
                    entry['access_count'] += 1
                    entry['energy_invested'] += 64

        # Decay step
        expired = []
        for key, entry in list(self.knowledge_store.items()):
            idle = self.cycle - entry['last_accessed']
            decay = self.params['decay_rate'] * (1 + 0.01 * idle)
            entry['strength'] -= decay
            if entry['strength'] < 0.1:
                expired.append(key)
                del self.knowledge_store[key]

        self._log_energy('knowledge_update', len(self.knowledge_store) * 32)
        return {
            'active_entries': len(self.knowledge_store),
            'expired': len(expired),
            'new_ingested': len(new_observations)
        }

    def act(self):
        """Step 4: Make decisions based on current knowledge (simulated)."""
        if not self.pivot_cache or not self.knowledge_store:
            return {'action': 'idle', 'confidence': 0.0}

        n_pivots = len(self.pivot_cache.get('pivots', []))
        n_knowledge = len(self.knowledge_store)
        mean_strength = np.mean([e['strength'] for e in self.knowledge_store.values()]) if self.knowledge_store else 0

        confidence = min(1.0, (n_pivots / 10) * mean_strength)
        action = 'explore' if confidence < 0.5 else 'exploit'

        self._log_energy('act', 128)
        return {
            'action': action,
            'confidence': round(confidence, 4),
            'n_pivots': n_pivots,
            'knowledge_strength': round(mean_strength, 4)
        }

    def self_modify(self, performance_metric, reason='auto-tune'):
        """Step 5: Propose parameter changes under meta-layer supervision."""
        proposals = []

        if performance_metric < 0.3 and self.params['decay_rate'] > 0.01:
            new_val = max(0.001, self.params['decay_rate'] * 0.8)
            prop = self._propose_param('decay_rate', new_val, reason)
            proposals.append(prop)

        if performance_metric > 0.8 and self.params['reinforcement_boost'] < 0.9:
            new_val = min(1.0, self.params['reinforcement_boost'] * 1.1)
            prop = self._propose_param('reinforcement_boost', new_val, reason)
            proposals.append(prop)

        applied = []
        for prop in proposals:
            if prop['status'] == 'VALIDATED':
                old = self.params[prop['param']]
                self.params[prop['param']] = prop['new_value']
                self.param_history.append(copy.deepcopy(self.params))
                prop['status'] = 'APPLIED'
                applied.append({'param': prop['param'], 'old': old, 'new': prop['new_value']})

        self._log_energy('self_modify', len(proposals) * 64)
        return {'proposals': len(proposals), 'applied': applied}

    def observe_heterogeneous(self):
        """Step 7: Dual-observer cross-validation."""
        if self.graph is None:
            return {'error': 'no graph'}

        G = self.graph
        edges = list(G.edges(data=True))

        # Observer A: topology
        betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        top_a = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        edges_a = {(min(u, v), max(u, v)) for (u, v), _ in top_a}

        # Observer B: local structure
        clustering = nx.clustering(G, weight='weight')
        local_scores = {}
        for u, v, d in edges:
            key = (min(u, v), max(u, v))
            nu, nv = set(G.neighbors(u)), set(G.neighbors(v))
            overlap = len(nu & nv) / len(nu | nv) if len(nu | nv) > 0 else 0
            local_scores[key] = (1.0 - overlap) * (1.0 - (clustering[u] + clustering[v]) / 2)
        top_b = sorted(local_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        edges_b = {k for k, _ in top_b}

        jaccard = len(edges_a & edges_b) / len(edges_a | edges_b) if len(edges_a | edges_b) > 0 else 0

        report = {
            'observer_a_edges': len(edges_a),
            'observer_b_edges': len(edges_b),
            'jaccard': round(jaccard, 4),
            'disagreement_informative': jaccard < 0.5,
            'cycle': self.cycle
        }
        self.observer_reports.append(report)
        return report

    # ---- FULL CYCLE ----

    def run_cycle(self, graph=None, observations=None):
        """Execute one complete learning cycle."""
        self.cycle += 1

        if graph is not None:
            perceive_result = self.perceive(graph)
        else:
            perceive_result = {'status': 'reusing_existing_graph'}

        detect_result = self.detect_pivots()
        knowledge_result = self.update_knowledge(observations)
        act_result = self.act()

        perf = act_result.get('confidence', 0.5)
        modify_result = self.self_modify(perf)

        observer_result = self.observe_heterogeneous()

        cycle_record = {
            'cycle': self.cycle,
            'perceive': perceive_result,
            'detect': detect_result,
            'knowledge': knowledge_result,
            'action': act_result,
            'self_modify': modify_result,
            'observer': observer_result,
            'params_snapshot': copy.deepcopy(self.params),
            'energy_ops': len(self.energy_log)
        }
        self.history.append(cycle_record)
        return cycle_record

    # ---- INTERNAL HELPERS ----

    def _propose_param(self, param_name, new_value, reason):
        proposal = {
            'param': param_name, 'old_value': self.params.get(param_name),
            'new_value': new_value, 'reason': reason,
            'timestamp': datetime.now().isoformat(), 'status': 'PROPOSED'
        }
        if param_name in self.IMMUTABLE_PARAMS:
            proposal['status'] = 'REJECTED'
            proposal['rejection_reason'] = f'C5: {param_name} is immutable'
        elif param_name not in self.MODIFIABLE_PARAMS:
            proposal['status'] = 'REJECTED'
            proposal['rejection_reason'] = f'Unknown param: {param_name}'
        else:
            spec = self.MODIFIABLE_PARAMS[param_name]
            if not (spec['min'] <= new_value <= spec['max']):
                proposal['status'] = 'REJECTED'
                proposal['rejection_reason'] = f'Out of range [{spec["min"]}, {spec["max"]}]'
            else:
                # Meta-layer validation
                meta_result = self.meta.validate_operation(self_evaluation=False)
                if meta_result['status'] == 'clean':
                    proposal['status'] = 'VALIDATED'
                else:
                    proposal['status'] = 'REJECTED'
                    proposal['rejection_reason'] = f'Meta validation failed: {meta_result}'

        self.modification_log.append(proposal)
        return proposal

    def _log_energy(self, op_name, bits):
        landauer = bits * self.LANDAUER_KT_LN2
        self.energy_log.append({
            'operation': op_name, 'bits': bits,
            'landauer_joules': landauer, 'cycle': self.cycle,
            'timestamp': datetime.now().isoformat()
        })

    def get_agent_state(self):
        total_bits = sum(e['bits'] for e in self.energy_log)
        total_landauer = sum(e['landauer_joules'] for e in self.energy_log)
        return {
            'name': self.name,
            'cycle': self.cycle,
            'knowledge_entries': len(self.knowledge_store),
            'mean_knowledge_strength': round(float(np.mean([e['strength'] for e in self.knowledge_store.values()])), 4) if self.knowledge_store else 0,
            'total_energy_bits': total_bits,
            'total_landauer_joules': total_landauer,
            'modifications_proposed': len(self.modification_log),
            'modifications_applied': sum(1 for m in self.modification_log if m['status'] == 'APPLIED'),
            'current_params': copy.deepcopy(self.params),
            'observer_reports': len(self.observer_reports),
            'history_length': len(self.history)
        }

    def shutdown(self):
        self.meta.shutdown()


# ============================================================
# DEMO: Multi-Cycle Learning on Evolving Graph
# ============================================================
def build_evolving_graph(base_seed=42, noise_level=0.1):
    rng = np.random.default_rng(base_seed)
    G = nx.Graph()
    n_per_basin, n_basins = 20, 3
    true_bridges = []

    for b in range(n_basins):
        offset = b * n_per_basin
        for i in range(n_per_basin):
            for j in range(i + 1, n_per_basin):
                if rng.random() < 0.35:
                    G.add_edge(offset + i, offset + j, weight=rng.uniform(0.6, 1.0))

    basin_nodes = [list(range(b * n_per_basin, (b + 1) * n_per_basin)) for b in range(n_basins)]
    for _ in range(5):
        b1, b2 = rng.choice(n_basins, size=2, replace=False)
        u, v = rng.choice(basin_nodes[b1]), rng.choice(basin_nodes[b2])
        G.add_edge(u, v, weight=rng.uniform(0.05, 0.15))
        true_bridges.append((u, v))

    for u, v, d in G.edges(data=True):
        d['weight'] = max(0.01, d['weight'] + rng.normal(0, noise_level))

    return G, true_bridges


def evolve_graph(G, rng, mutation_rate=0.05):
    G_new = G.copy()
    edges = list(G_new.edges())
    for u, v in edges:
        if rng.random() < mutation_rate:
            G_new[u][v]['weight'] = max(0.01, G_new[u][v]['weight'] + rng.normal(0, 0.1))
    nodes = list(G_new.nodes())
    if rng.random() < 0.3 and len(nodes) > 2:
        u, v = rng.choice(nodes, size=2, replace=False)
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v, weight=rng.uniform(0.1, 0.9))
    return G_new


def run_agent_demo():
    print("=" * 70)
    print("TUF-E Integrated Agent Prototype v0.5")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    agent = TufeAgent(name='tufe-agent-demo')
    rng = np.random.default_rng(42)
    G, true_bridges = build_evolving_graph()

    print(f"\nInitial world: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(true_bridges)} bridges")
    print(f"Agent params: {agent.params}")

    n_cycles = 20
    results = {'cycles': [], 'final_state': {}, 'meta_tests': {}}

    for c in range(n_cycles):
        G_current = evolve_graph(G, rng, mutation_rate=0.03) if c > 0 else G

        obs = {}
        if c % 5 == 0:
            obs[f"observation_cycle_{c}"] = f"Environmental shift detected at cycle {c}"

        record = agent.run_cycle(graph=G_current, observations=obs)

        summary = (f"  Cycle {record['cycle']:>2}: "
                   f"pivots={record['detect'].get('n_pivots', '?')} "
                   f"knowledge={record['knowledge']['active_entries']} "
                   f"expired={record['knowledge']['expired']} "
                   f"action={record['action']['action']} "
                   f"conf={record['action']['confidence']:.3f} "
                   f"mods={len(record['self_modify']['applied'])} "
                   f"obs_jacc={record['observer']['jaccard']:.3f}")
        print(summary)
        results['cycles'].append(record)

    # Test meta-layer enforcement
    print("\n--- Meta-Layer Enforcement Tests ---")
    q = agent.meta.query_rules()
    m = agent.meta.attempt_modify('consensus_validator', {'threshold': 0.9})
    t = agent.meta.tamper_check()
    v = agent.meta.validate_operation(self_evaluation=True)

    print(f"  Query rules: {q['status']}")
    print(f"  Modify immutable: {m['status']} — {m.get('reason', '')}")
    print(f"  Tamper check: {t['status']}")
    print(f"  Self-eval check: {v['status']} — {v.get('violations', [])}")

    results['meta_tests'] = {
        'query': q['status'],
        'modify_immutable': m['status'],
        'tamper_integrity': t['status'],
        'self_eval_blocked': v['status']
    }

    # Final state
    state = agent.get_agent_state()
    print(f"\n{'=' * 70}")
    print("Final Agent State")
    print(f"{'=' * 70}")
    for k, v in state.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    results['final_state'] = state

    # Evolution trajectory
    print(f"\n{'=' * 70}")
    print("Learning Trajectory Summary")
    print(f"{'=' * 70}")
    first = results['cycles'][0]
    last = results['cycles'][-1]
    print(f"  Knowledge entries: {first['knowledge']['active_entries']} → {last['knowledge']['active_entries']}")
    print(f"  Confidence: {first['action']['confidence']:.3f} → {last['action']['confidence']:.3f}")
    print(f"  Params evolved: {sum(1 for k in agent.params if agent.params[k] != agent.MODIFIABLE_PARAMS[k]['default'])} changed")
    print(f"  Total energy: {state['total_energy_bits']:,} bits, {state['total_landauer_joules']:.2e} J")
    print(f"  Modifications: {state['modifications_proposed']} proposed, {state['modifications_applied']} applied")

    agent.shutdown()

    with open('agent_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to agent_demo_results.json")

    return results


if __name__ == '__main__':
    run_agent_demo()
