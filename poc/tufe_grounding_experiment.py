"""
TUF-E Semantic Grounding Experiment v0.1

核心问题: 在没有预定义语义的情况下，智能体能否从因果反馈中
自发涌现出有意义的操作结构？

环境: 算术表达式求值器
  - 智能体不知道 +、-、*、/ 的含义
  - 它只知道: 执行某个操作序列 → 得到某个结果
  - 目标: 从输入输出对中学会正确的操作组合

智能体的学习过程:
  1. 观察 (input, output) 对
  2. 尝试操作序列（随机探索 → 逐渐聚焦）
  3. 成功/失败作为能量信号注入耗散知识库
  4. 枢机检测识别哪些操作是"不可绕过的"
  5. 成功的操作组合被强化，失败的衰减遗忘
  6. 多轮后观察: 是否涌现出对应真实运算的结构？

这不是教智能体做算术。这是观察: 在有因果反馈的环境中，
TUF-E约束体系是否会自发产生有意义的知识结构。
"""

import numpy as np
import networkx as nx
from ripser import ripser
import json
import time
import sys
import io
import copy
import heapq
import hashlib
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
import itertools
import operator

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# ENVIRONMENT: Arithmetic Expression Evaluator
# ============================================================
class ArithmeticEnvironment:
    """
    简单的算术世界。

    智能体看到的: (operands, target_result)
    智能体做的: 选择操作序列
    环境反馈: success/fail + actual_result

    可用操作(智能体不知道它们的含义):
      op_0: add       op_1: subtract    op_2: multiply    op_3: divide
      op_4: negate    op_5: abs         op_6: square      op_7: identity

    智能体只看到操作ID和数值结果，不看到操作名称。
    """

    OPS = {
        0: ('add', lambda a, b: a + b),
        1: ('sub', lambda a, b: a - b),
        2: ('mul', lambda a, b: a * b),
        3: ('div', lambda a, b: a / b if b != 0 else float('inf')),
        4: ('neg', lambda a, b: -a),
        5: ('abs', lambda a, b: abs(a)),
        6: ('sqr', lambda a, b: a * a),
        7: ('id', lambda a, b: a),
    }

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.history = []

    def generate_task(self, difficulty=1):
        """生成一个算术任务: (operands, target, solution_ops)"""
        if difficulty == 1:
            a = int(self.rng.integers(1, 20))
            b = int(self.rng.integers(1, 20))
            op_id = int(self.rng.choice([0, 1, 2]))  # add, sub, mul only
            _, op_fn = self.OPS[op_id]
            target = op_fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id]}
        elif difficulty == 2:
            a = int(self.rng.integers(1, 10))
            b = int(self.rng.integers(1, 10))
            c = int(self.rng.integers(1, 10))
            op1_id = int(self.rng.choice([0, 1, 2]))
            op2_id = int(self.rng.choice([0, 1, 2]))
            _, op1_fn = self.OPS[op1_id]
            _, op2_fn = self.OPS[op2_id]
            intermediate = op1_fn(a, b)
            target = op2_fn(intermediate, c)
            return {'operands': [a, b, c], 'target': target, 'solution_ops': [op1_id, op2_id]}
        else:
            a = int(self.rng.integers(1, 10))
            b = int(self.rng.integers(1, 10))
            op_id = int(self.rng.choice([0, 1, 2, 6]))
            _, op_fn = self.OPS[op_id]
            target = op_fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id]}

    def execute(self, operands, op_sequence):
        """执行操作序列，返回结果"""
        try:
            values = list(operands)
            for op_id in op_sequence:
                if op_id not in self.OPS:
                    return {'success': False, 'result': None, 'error': 'unknown_op'}
                _, op_fn = self.OPS[op_id]
                if len(values) >= 2:
                    result = op_fn(values[0], values[1])
                    values = [result] + values[2:]
                else:
                    result = op_fn(values[0], 0)
                    values = [result]
            return {'success': True, 'result': values[0] if values else None}
        except Exception as e:
            return {'success': False, 'result': None, 'error': str(e)}

    def evaluate(self, task, op_sequence):
        """评估智能体的操作序列是否正确"""
        exec_result = self.execute(task['operands'], op_sequence)
        if not exec_result['success']:
            return {'correct': False, 'feedback': 'execution_error', 'result': None}
        result = exec_result['result']
        target = task['target']
        correct = abs(result - target) < 0.001 if isinstance(result, (int, float)) else False
        return {
            'correct': correct,
            'feedback': 'success' if correct else 'wrong_result',
            'result': result,
            'target': target
        }


# ============================================================
# TUF-E LEARNING AGENT (Semantic Grounding Version)
# ============================================================
class GroundedTufeAgent:
    """
    接地版TUF-E智能体。

    与之前原型的区别:
    - 知识不是抽象的key-value，而是操作序列→结果的因果记录
    - 枢机检测在操作图上运行，识别不可绕过的操作步骤
    - 能量信号来自环境的success/fail反馈，不是模拟的
    - 自修改改变的是操作选择策略，不只是数值参数
    """

    N_OPS = 8

    def __init__(self, name='grounded-agent-v0.1'):
        self.name = name
        self.cycle = 0

        # Operation graph: nodes=operations, edges=co-occurrence in successful sequences
        self.op_graph = nx.Graph()
        for i in range(self.N_OPS):
            self.op_graph.add_node(i, use_count=0, success_count=0)

        # Knowledge store: (operand_pattern, op_seq) -> {strength, outcomes, ...}
        # Key includes operand count to enable generalization
        self.knowledge = {}

        # Per-operand-count action preferences
        self.op_preferences_by_arity = {
            2: np.ones(self.N_OPS) / self.N_OPS,
            3: np.ones(self.N_OPS) / self.N_OPS,
        }

        # Parameters
        self.params = {
            'exploration_rate': 0.4,
            'decay_rate': 0.02,
            'reinforcement_boost': 0.4,
            'min_viability': 0.05,
            'proximity_reward_threshold': 0.3,  # reward near-misses
        }

        # Statistics
        self.total_attempts = 0
        self.total_successes = 0
        self.energy_bits = 0
        self.history = []
        self.rng = np.random.default_rng(42)

    def choose_actions(self, n_operands, explore_override=None):
        """选择操作序列（探索 vs 利用），利用任务结构"""
        exploration = explore_override if explore_override is not None else self.params['exploration_rate']

        # Determine sequence length from operand count (structural hint)
        seq_len = max(1, n_operands - 1)
        prefs = self.op_preferences_by_arity.get(n_operands, np.ones(self.N_OPS) / self.N_OPS)

        if self.rng.random() < exploration:
            ops = [int(self.rng.choice(self.N_OPS, p=prefs)) for _ in range(seq_len)]
        else:
            # Exploit: find best matching knowledge
            arity_key = n_operands
            candidates = [(k, v) for k, v in self.knowledge.items()
                          if k[0] == arity_key and v['strength'] > self.params['min_viability']]
            if candidates:
                best = max(candidates, key=lambda x: x[1]['strength'])
                ops = list(best[0][1])  # (arity, op_seq) -> op_seq
                # Pad or trim to match current need
                while len(ops) < seq_len:
                    ops.append(int(self.rng.choice(self.N_OPS, p=prefs)))
                ops = ops[:seq_len]
            else:
                ops = [int(self.rng.choice(self.N_OPS, p=prefs)) for _ in range(seq_len)]

        return ops

    def rng_choice(self):
        return np.random.random()

    def learn_from_feedback(self, task, op_sequence, feedback):
        """从环境反馈中学习，包含近似奖励"""
        self.cycle += 1
        self.total_attempts += 1
        n_operands = len(task['operands'])
        seq_key = (n_operands, tuple(op_sequence))

        bits = len(op_sequence) * 32 + 64
        self.energy_bits += bits

        # Proximity reward: even wrong answers get partial credit if close
        is_correct = feedback['correct']
        is_close = False
        if not is_correct and feedback.get('result') is not None and feedback.get('target') is not None:
            try:
                ratio = abs(feedback['result'] - feedback['target']) / max(abs(feedback['target']), 1)
                is_close = ratio < self.params['proximity_reward_threshold']
            except (TypeError, ZeroDivisionError):
                pass

        prefs = self.op_preferences_by_arity.setdefault(n_operands, np.ones(self.N_OPS) / self.N_OPS)

        if is_correct:
            self.total_successes += 1
            reward = 1.0
        elif is_close:
            reward = 0.3
        else:
            reward = 0.0

        if reward > 0:
            if seq_key in self.knowledge:
                entry = self.knowledge[seq_key]
                entry['strength'] = min(1.0, entry['strength'] + self.params['reinforcement_boost'] * reward)
                entry['successes'] += int(is_correct)
                entry['near_misses'] += int(is_close and not is_correct)
                entry['last_used'] = self.cycle
            else:
                self.knowledge[seq_key] = {
                    'strength': 0.3 * reward + self.params['reinforcement_boost'] * reward,
                    'successes': int(is_correct),
                    'near_misses': int(is_close and not is_correct),
                    'failures': 0,
                    'created': self.cycle,
                    'last_used': self.cycle,
                    'operands_pattern': n_operands,
                }

            for i in range(len(op_sequence)):
                op = op_sequence[i]
                self.op_graph.nodes[op]['use_count'] += 1
                self.op_graph.nodes[op]['success_count'] += 1
                for j in range(i + 1, len(op_sequence)):
                    op2 = op_sequence[j]
                    if self.op_graph.has_edge(op, op2):
                        self.op_graph[op][op2]['weight'] += 1.0
                    else:
                        self.op_graph.add_edge(op, op2, weight=1.0)

            for op in op_sequence:
                prefs[op] *= (1.0 + 0.15 * reward)
            prefs /= prefs.sum()
        else:
            if seq_key in self.knowledge:
                self.knowledge[seq_key]['strength'] -= self.params['decay_rate']
                self.knowledge[seq_key]['failures'] += 1
                if self.knowledge[seq_key]['strength'] < self.params['min_viability']:
                    del self.knowledge[seq_key]

            for op in op_sequence:
                prefs[op] *= 0.97
            prefs /= prefs.sum()

        # Decay step
        expired = []
        for key, entry in list(self.knowledge.items()):
            idle = self.cycle - entry['last_used']
            decay = self.params['decay_rate'] * (1 + 0.002 * idle)
            entry['strength'] -= decay
            if entry['strength'] < self.params['min_viability']:
                expired.append(key)
                del self.knowledge[key]

        return {
            'correct': is_correct,
            'close': is_close,
            'knowledge_size': len(self.knowledge),
            'expired': len(expired),
            'energy_bits': bits
        }

    def detect_pivots(self):
        """在操作图上检测枢机：哪些操作是不可绕过的？"""
        if self.op_graph.number_of_edges() == 0:
            return {'pivots': [], 'method': 'none'}

        G = self.op_graph.copy()

        # Category-theoretic: find irreducible edges
        adj = {n: {} for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            adj[u][v] = d.get('weight', 1.0)
            adj[v][u] = d.get('weight', 1.0)

        pivot_ops = {}
        for u, v, d in G.edges(data=True):
            w_direct = d.get('weight', 1.0)
            # Check if there's an alternative path
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
            pivot_ops[(u, v)] = irr

        # Rank by irreducibility
        ranked = sorted(pivot_ops.items(), key=lambda x: x[1], reverse=True)
        pivots = [(e, s) for e, s in ranked if s > 0.5]

        # Also identify high-success operations as structural pivots
        node_pivots = []
        for node, data in G.nodes(data=True):
            if data.get('success_count', 0) > 0:
                success_rate = data['success_count'] / max(data['use_count'], 1)
                if success_rate > 0.5:
                    node_pivots.append((node, success_rate, data['success_count']))

        return {
            'edge_pivots': pivots[:5],
            'node_pivots': sorted(node_pivots, key=lambda x: x[1], reverse=True)[:5],
            'n_edges': G.number_of_edges(),
            'method': 'category_theoretic_local_k'
        }

    def get_state(self):
        success_rate = self.total_successes / max(self.total_attempts, 1)
        # Merge preferences across arities for reporting
        all_prefs = np.zeros(self.N_OPS)
        count = 0
        for prefs in self.op_preferences_by_arity.values():
            all_prefs += prefs
            count += 1
        if count > 0:
            all_prefs /= count
        return {
            'cycle': self.cycle,
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'success_rate': round(success_rate, 4),
            'knowledge_entries': len(self.knowledge),
            'op_graph_edges': self.op_graph.number_of_edges(),
            'op_preferences': {i: round(float(p), 4) for i, p in enumerate(all_prefs)},
            'energy_bits': self.energy_bits,
            'top_knowledge': [
                {'ops': list(k[1]), 'arity': k[0], 'strength': round(v['strength'], 3), 'successes': v['successes']}
                for k, v in sorted(self.knowledge.items(), key=lambda x: x[1]['strength'], reverse=True)[:5]
            ]
        }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_grounding_experiment(n_episodes=500, report_every=50):
    print("=" * 70)
    print("TUF-E Semantic Grounding Experiment v0.1")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    env = ArithmeticEnvironment(seed=42)
    agent = GroundedTufeAgent()

    results = {
        'episodes': [],
        'milestones': [],
        'final_state': {},
        'pivot_analysis': {},
        'emergent_structure': {}
    }

    print(f"\nEnvironment: arithmetic expressions, {GroundedTufeAgent.N_OPS} operations")
    print(f"Episodes: {n_episodes}, reporting every {report_every}")
    print(f"Initial exploration rate: {agent.params['exploration_rate']}")

    phase_transitions = []

    for ep in range(n_episodes):
        # Difficulty increases over time
        if ep < n_episodes * 0.4:
            difficulty = 1
        elif ep < n_episodes * 0.7:
            difficulty = 2
        else:
            difficulty = 3

        task = env.generate_task(difficulty=difficulty)

        # Agent chooses actions
        ops = agent.choose_actions(len(task['operands']))

        # Environment evaluates
        feedback = env.evaluate(task, ops)

        # Agent learns
        learn_result = agent.learn_from_feedback(task, ops, feedback)

        record = {
            'episode': ep,
            'difficulty': difficulty,
            'task_ops': task['operands'],
            'target': task['target'],
            'chosen_ops': ops,
            'correct': feedback['correct'],
            'knowledge_size': learn_result['knowledge_size'],
            'success_rate_so_far': agent.total_successes / max(agent.total_attempts, 1)
        }
        results['episodes'].append(record)

        # Report progress
        if (ep + 1) % report_every == 0:
            state = agent.get_state()
            sr = state['success_rate']
            ks = state['knowledge_entries']
            ge = state['op_graph_edges']
            top_ops = sorted(state['op_preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
            top_op_names = [f"op_{k}({ArithmeticEnvironment.OPS[k][0]})={v:.3f}" for k, v in top_ops]

            print(f"  Ep {ep+1:>3}/{n_episodes}: "
                  f"SR={sr:.3f} | "
                  f"knowledge={ks} | "
                  f"graph_edges={ge} | "
                  f"top_ops: {', '.join(top_op_names)}")

            results['milestones'].append(state)

            # Detect phase transitions (significant jumps in success rate)
            if len(results['milestones']) >= 2:
                prev_sr = results['milestones'][-2]['success_rate']
                if sr - prev_sr > 0.1:
                    phase_transitions.append({'episode': ep, 'from': prev_sr, 'to': sr})

    # Final analysis
    final_state = agent.get_state()
    results['final_state'] = final_state

    print(f"\n{'=' * 70}")
    print("Final State")
    print(f"{'=' * 70}")
    print(f"  Success rate: {final_state['success_rate']:.3f} ({final_state['total_successes']}/{final_state['total_attempts']})")
    print(f"  Knowledge entries: {final_state['knowledge_entries']}")
    print(f"  Op graph edges: {final_state['op_graph_edges']}")
    print(f"  Energy consumed: {final_state['energy_bits']:,} bits")
    print(f"  Operation preferences:")
    for i in range(GroundedTufeAgent.N_OPS):
        name = ArithmeticEnvironment.OPS[i][0]
        pref = final_state['op_preferences'][i]
        bar = '#' * int(pref * 50)
        print(f"    op_{i} ({name:>3}): {pref:.4f} {bar}")

    # Pivot analysis
    print(f"\n--- Pivot Analysis ---")
    pivots = agent.detect_pivots()
    print(f"  Edge pivots (irreducible operation pairs):")
    for (u, v), score in pivots.get('edge_pivots', []):
        name_u = ArithmeticEnvironment.OPS[u][0]
        name_v = ArithmeticEnvironment.OPS[v][0]
        print(f"    op_{u}({name_u}) ↔ op_{v}({name_v}): irreducibility={score:.3f}")

    print(f"  Node pivots (high-success operations):")
    for node, rate, count in pivots.get('node_pivots', []):
        name = ArithmeticEnvironment.OPS[node][0]
        print(f"    op_{node}({name}): success_rate={rate:.3f}, uses={count}")

    results['pivot_analysis'] = pivots

    # Emergent structure analysis
    print(f"\n--- Emergent Structure ---")
    top_knowledge = final_state.get('top_knowledge', [])
    print(f"  Top learned operation sequences:")
    for entry in top_knowledge:
        op_names = [ArithmeticEnvironment.OPS[o][0] for o in entry['ops']]
        print(f"    {entry['ops']} ({'+'.join(op_names)}): strength={entry['strength']:.3f}, successes={entry['successes']}")

    # Check if the agent discovered the correct operations
    preferred_ops = sorted(final_state['op_preferences'].items(), key=lambda x: x[1], reverse=True)
    top3_ids = [k for k, v in preferred_ops[:3]]
    correct_ops = {0, 1, 2}  # add, sub, mul are the solution ops for difficulty 1-2
    discovered = len(set(top3_ids) & correct_ops)
    print(f"\n  Discovery metric: {discovered}/3 core operations in top-3 preferences")
    print(f"  Phase transitions detected: {len(phase_transitions)}")
    for pt in phase_transitions:
        print(f"    Episode {pt['episode']}: SR {pt['from']:.3f} → {pt['to']:.3f}")

    results['emergent_structure'] = {
        'core_ops_discovered': discovered,
        'phase_transitions': phase_transitions,
        'top_knowledge_sequences': top_knowledge
    }

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    with open('grounding_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_experiment_results.json")

    return results


if __name__ == '__main__':
    run_grounding_experiment(n_episodes=500, report_every=50)
