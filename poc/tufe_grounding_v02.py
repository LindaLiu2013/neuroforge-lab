"""
TUF-E Grounded Agent v0.2: 有损自感知 + 可组合枢机库

v0.1的问题:
  - 智能体可以直接读取自己的操作图和知识库(违反C5精神)
  - 知识是扁平的key-value，不可组合
  - 环境太简单，无延迟反馈、无部分可观察

v0.2改进:
  1. 有损自感知: 智能体只能通过采样探测了解自己的状态
     - 每次探测消耗能量预算
     - 探测结果是局部的、有噪声的
     - 不能直接读取偏好向量或知识库
  2. 可组合枢机库: 成功序列被分解为最小因果单元
     - 枢机 = (输入模式, 操作, 输出模式)三元组
     - 枢机可以拼接: 如果A→B和B→C都是枢机，可以尝试A→B→C
     - 拼接结果由环境验证
  3. 升级环境:
     - 多步任务(需要2-3步操作)
     - 部分可观察(只看到输入的某些特征)
     - 延迟反馈(每N步才给一次成功/失败信号)
"""

import numpy as np
import networkx as nx
import json
import sys
import io
import heapq
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# UPGRADED ENVIRONMENT
# ============================================================
class UpgradedEnvironment:
    """
    升级版算术环境。

    特点:
    - 任务需要1-3步操作
    - 智能体只看到输入的部分特征(数值范围，不是精确值)
    - 反馈有延迟(每feedback_delay步才给一次)
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
    N_OPS = 8

    def __init__(self, seed=42, feedback_delay=3):
        self.rng = np.random.default_rng(seed)
        self.feedback_delay = feedback_delay
        self.pending_feedback = []

    def observe_partial(self, operands):
        """部分可观察: 只返回数值的模糊特征"""
        features = []
        for v in operands:
            av = abs(v)
            if av < 5:
                feat = 'tiny'
            elif av < 15:
                feat = 'small'
            elif av < 50:
                feat = 'medium'
            else:
                feat = 'large'
            sign = 'pos' if v >= 0 else 'neg'
            even = 'even' if v % 2 == 0 else 'odd'
            features.append({'magnitude': feat, 'sign': sign, 'parity': even})
        return features

    def generate_task(self, difficulty=1):
        if difficulty <= 1:
            a = int(self.rng.integers(1, 20))
            b = int(self.rng.integers(1, 20))
            op_id = int(self.rng.choice([0, 1, 2]))
            _, fn = self.OPS[op_id]
            target = fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id], 'steps': 1}
        elif difficulty == 2:
            a = int(self.rng.integers(1, 10))
            b = int(self.rng.integers(1, 10))
            c = int(self.rng.integers(1, 10))
            op1 = int(self.rng.choice([0, 1, 2]))
            op2 = int(self.rng.choice([0, 1, 2]))
            _, fn1 = self.OPS[op1]
            _, fn2 = self.OPS[op2]
            mid = fn1(a, b)
            target = fn2(mid, c)
            return {'operands': [a, b, c], 'target': target, 'solution_ops': [op1, op2], 'steps': 2}
        else:
            a = int(self.rng.integers(1, 8))
            b = int(self.rng.integers(1, 8))
            op_id = int(self.rng.choice([0, 1, 2, 6]))
            _, fn = self.OPS[op_id]
            target = fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id], 'steps': 1}

    def execute(self, operands, op_sequence):
        try:
            values = list(operands)
            for op_id in op_sequence:
                if op_id not in self.OPS:
                    return None
                _, fn = self.OPS[op_id]
                if len(values) >= 2:
                    result = fn(values[0], values[1])
                    values = [result] + values[2:]
                else:
                    values = [fn(values[0], 0)]
            return values[0] if values else None
        except Exception:
            return None

    def submit_attempt(self, task, op_sequence, episode):
        """提交尝试，反馈可能延迟"""
        result = self.execute(task['operands'], op_sequence)
        correct = False
        if result is not None:
            try:
                correct = abs(result - task['target']) < 0.001
            except TypeError:
                pass

        entry = {
            'episode': episode,
            'correct': correct,
            'result': result,
            'target': task['target'],
            'ops': op_sequence
        }
        self.pending_feedback.append(entry)

        if len(self.pending_feedback) >= self.feedback_delay:
            batch = self.pending_feedback[:self.feedback_delay]
            self.pending_feedback = self.pending_feedback[self.feedback_delay:]
            return batch
        return None

    def flush_feedback(self):
        batch = list(self.pending_feedback)
        self.pending_feedback = []
        return batch


# ============================================================
# COMPOSABLE PIVOT LIBRARY
# ============================================================
class PivotLibrary:
    """
    可组合枢机库。

    枢机 = (input_pattern, op_id, output_pattern)
    input/output_pattern是数值的离散化描述(magnitude+sign)

    组合规则: 如果pivot_A.output ≈ pivot_B.input，
    则可以尝试 [pivot_A.ops..., pivot_B.ops...] 作为组合枢机。
    """

    def __init__(self, min_strength=0.1):
        self.pivots = {}  # key: (in_pat, op, out_pat) -> {strength, uses, successes}
        self.compositions = {}  # key: tuple of pivot keys -> {strength, uses}
        self.min_strength = min_strength

    @staticmethod
    def discretize(value):
        av = abs(value) if value is not None else 0
        if av < 5:
            mag = 'tiny'
        elif av < 15:
            mag = 'small'
        elif av < 50:
            mag = 'medium'
        else:
            mag = 'large'
        sign = 'pos' if (value or 0) >= 0 else 'neg'
        return f"{mag}_{sign}"

    def record(self, input_val, op_id, output_val, success):
        in_pat = self.discretize(input_val)
        out_pat = self.discretize(output_val)
        key = (in_pat, op_id, out_pat)

        if key in self.pivots:
            p = self.pivots[key]
            p['uses'] += 1
            if success:
                p['successes'] += 1
                p['strength'] = min(1.0, p['strength'] + 0.2)
            else:
                p['strength'] = max(0, p['strength'] - 0.05)
        else:
            self.pivots[key] = {
                'strength': 0.3 if success else 0.05,
                'uses': 1,
                'successes': 1 if success else 0,
                'ops': [op_id]
            }

    def find_composable_pairs(self):
        """找output≈input匹配的枢机对"""
        pairs = []
        pivot_list = [(k, v) for k, v in self.pivots.items()
                      if v['strength'] >= self.min_strength]
        for (in_a, op_a, out_a), pa in pivot_list:
            for (in_b, op_b, out_b), pb in pivot_list:
                if out_a == in_b:  # output of A matches input of B
                    pair_key = ((in_a, op_a, out_a), (in_b, op_b, out_b))
                    pairs.append({
                        'key': pair_key,
                        'combined_ops': pa['ops'] + pb['ops'],
                        'strength': min(pa['strength'], pb['strength']),
                        'in_pattern': in_a,
                        'out_pattern': out_b
                    })
        pairs.sort(key=lambda x: x['strength'], reverse=True)
        return pairs

    def record_composition(self, comp_key, success):
        if comp_key in self.compositions:
            c = self.compositions[comp_key]
            c['uses'] += 1
            if success:
                c['strength'] = min(1.0, c['strength'] + 0.15)
                c['successes'] += 1
            else:
                c['strength'] = max(0, c['strength'] - 0.03)
        else:
            self.compositions[comp_key] = {
                'strength': 0.2 if success else 0.02,
                'uses': 1,
                'successes': 1 if success else 0
            }

    def decay(self, rate=0.01):
        expired_p = []
        for k, p in list(self.pivots.items()):
            p['strength'] -= rate
            if p['strength'] < self.min_strength:
                expired_p.append(k)
                del self.pivots[k]

        expired_c = []
        for k, c in list(self.compositions.items()):
            c['strength'] -= rate
            if c['strength'] < self.min_strength:
                expired_c.append(k)
                del self.compositions[k]

        return len(expired_p), len(expired_c)

    def get_summary(self):
        return {
            'n_pivots': len(self.pivots),
            'n_compositions': len(self.compositions),
            'mean_pivot_strength': round(float(np.mean([p['strength'] for p in self.pivots.values()])), 3) if self.pivots else 0,
            'top_pivots': sorted(
                [{'pattern': f"{k[0]}→op_{k[1]}→{k[2]}", 'strength': round(v['strength'], 3),
                  'successes': v['successes'], 'uses': v['uses']}
                 for k, v in self.pivots.items()],
                key=lambda x: x['strength'], reverse=True
            )[:8],
            'top_compositions': sorted(
                [{'ops': v.get('ops', []), 'strength': round(v['strength'], 3)}
                 for v in self.compositions.values()],
                key=lambda x: x['strength'], reverse=True
            )[:5]
        }


# ============================================================
# GROUNDED AGENT v0.2
# ============================================================
class GroundedAgentV02:
    """
    v0.2智能体: 有损自感知 + 可组合枢机库
    """

    def __init__(self, probe_budget_per_cycle=3, name='grounded-v0.2'):
        self.name = name
        self.cycle = 0
        self.probe_budget = probe_budget_per_cycle
        self.probes_used = 0

        # Internal state (NOT directly accessible to decision-making)
        self._internal_op_graph = nx.Graph()
        for i in range(UpgradedEnvironment.N_OPS):
            self._internal_op_graph.add_node(i, use_count=0, success_count=0)
        self._internal_preferences = {
            2: np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS,
            3: np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS,
        }

        # Pivot library (accessible only through probing)
        self.pivot_lib = PivotLibrary()

        # What the agent "knows" about itself (lossy view)
        self.self_model = {
            'estimated_preferences': np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS,
            'known_pivots': [],
            'known_compositions': [],
            'confidence': 0.0,
            'last_probe_cycle': -1
        }

        # Stats
        self.total_attempts = 0
        self.total_successes = 0
        self.energy_bits = 0
        self.history = []
        self.rng = np.random.default_rng(42)

    def probe_self(self, n_probes=None):
        """
        有损自感知: 通过采样探测了解自己的状态。
        每次探测消耗能量，结果是噪声的。
        """
        if n_probes is None:
            n_probes = self.probe_budget

        actual_probes = min(n_probes, self.probe_budget - self.probes_used)
        if actual_probes <= 0:
            return self.self_model

        self.probes_used += actual_probes
        self.energy_bits += actual_probes * 128

        # Probe 1: Sample operation preferences with noise
        noisy_prefs = np.zeros(UpgradedEnvironment.N_OPS)
        for _ in range(actual_probes):
            # Pick a random arity and sample from true preferences
            arity = self.rng.choice([2, 3])
            true_prefs = self._internal_preferences.get(arity, np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS)
            sampled_op = self.rng.choice(UpgradedEnvironment.N_OPS, p=true_prefs)
            noisy_prefs[sampled_op] += 1
        if noisy_prefs.sum() > 0:
            noisy_prefs /= noisy_prefs.sum()
        else:
            noisy_prefs = np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS

        # Add observation noise
        noise = self.rng.dirichlet(np.ones(UpgradedEnvironment.N_OPS) * 10)
        noisy_prefs = 0.7 * noisy_prefs + 0.3 * noise

        self.self_model['estimated_preferences'] = noisy_prefs
        self.self_model['last_probe_cycle'] = self.cycle

        # Probe 2: Sample known pivots (only see top-k with some randomness)
        all_pivots = list(self.pivot_lib.pivots.items())
        if all_pivots:
            k = min(actual_probes, len(all_pivots))
            # Weighted sampling by strength
            strengths = np.array([v['strength'] for _, v in all_pivots])
            if strengths.sum() > 0:
                probs = strengths / strengths.sum()
                indices = self.rng.choice(len(all_pivots), size=k, replace=False, p=probs) if k <= len(all_pivots) else self.rng.choice(len(all_pivots), size=k, replace=True, p=probs)
                sampled = [all_pivots[i] for i in indices]
                self.self_model['known_pivots'] = [
                    {'pattern': f"{k_[0]}→op_{k_[1]}→{k_[2]}", 'strength': round(v['strength'], 3)}
                    for k_, v in sampled
                ]
            else:
                self.self_model['known_pivots'] = []

        # Probe 3: Check compositions
        comps = self.pivot_lib.find_composable_pairs()
        if comps:
            self.self_model['known_compositions'] = [
                {'ops': c['combined_ops'], 'strength': round(c['strength'], 3)}
                for c in comps[:actual_probes]
            ]

        # Confidence based on how recently we probed and how many probes
        self.self_model['confidence'] = min(1.0, actual_probes / self.probe_budget * 0.8)

        return self.self_model

    def choose_actions(self, observation, n_operands):
        """基于有损自模型选择操作"""
        seq_len = max(1, n_operands - 1)
        prefs = self.self_model['estimated_preferences']

        # Exploration vs exploitation based on self-confidence
        exploration = max(0.1, 1.0 - self.self_model['confidence'])

        # Try to use known compositions first
        if self.rng.random() > exploration and self.self_model['known_compositions']:
            best_comp = max(self.self_model['known_compositions'], key=lambda x: x['strength'])
            ops = best_comp['ops'][:seq_len]
            while len(ops) < seq_len:
                ops.append(int(self.rng.choice(UpgradedEnvironment.N_OPS, p=prefs)))
            return ops[:seq_len]

        # Try known pivots
        if self.rng.random() > exploration and self.self_model['known_pivots']:
            # Match pivot input pattern to observation
            obs_mag = observation[0]['magnitude'] if observation else 'medium'
            matching = [p for p in self.self_model['known_pivots'] if obs_mag in p['pattern']]
            if matching:
                best = max(matching, key=lambda x: x['strength'])
                # Extract op from pattern string
                try:
                    op_str = best['pattern'].split('→')[1]
                    op_id = int(op_str.replace('op_', ''))
                    return [op_id] * seq_len
                except (IndexError, ValueError):
                    pass

        # Fall back to preference-based selection
        if self.rng.random() < exploration:
            return [int(self.rng.choice(UpgradedEnvironment.N_OPS)) for _ in range(seq_len)]
        else:
            return [int(self.rng.choice(UpgradedEnvironment.N_OPS, p=prefs)) for _ in range(seq_len)]

    def learn_from_feedback_batch(self, feedback_batch, tasks_info):
        """从延迟反馈批次中学习"""
        self.cycle += 1
        self.probes_used = 0  # Reset probe budget each cycle

        for fb in feedback_batch:
            self.total_attempts += 1
            ops = fb['ops']
            correct = fb['correct']
            result = fb['result']
            target = fb['target']

            if correct:
                self.total_successes += 1

            bits = len(ops) * 32 + 64
            self.energy_bits += bits

            # Update internal state (not directly visible to agent)
            arity = len(ops) + 1
            prefs = self._internal_preferences.setdefault(arity, np.ones(UpgradedEnvironment.N_OPS) / UpgradedEnvironment.N_OPS)

            if correct:
                for op in ops:
                    prefs[op] *= 1.15
                    self._internal_op_graph.nodes[op]['use_count'] += 1
                    self._internal_op_graph.nodes[op]['success_count'] += 1
                prefs /= prefs.sum()

                # Strengthen co-occurrence edges
                for i in range(len(ops)):
                    for j in range(i + 1, len(ops)):
                        if self._internal_op_graph.has_edge(ops[i], ops[j]):
                            self._internal_op_graph[ops[i]][ops[j]]['weight'] += 1.0
                        else:
                            self._internal_op_graph.add_edge(ops[i], ops[j], weight=1.0)
            else:
                for op in ops:
                    prefs[op] *= 0.97
                prefs /= prefs.sum()

            # Record in pivot library
            # For each step in the sequence, record input→op→output
            values = tasks_info.get(fb['episode'], {}).get('operands', [0, 0])
            current_val = values[0] if values else 0
            for i, op in enumerate(ops):
                next_val = result if i == len(ops) - 1 else current_val
                self.pivot_lib.record(current_val, op, next_val, correct)
                if i < len(values) - 1:
                    current_val = values[i + 1]

            # Try composition learning for multi-step sequences
            if len(ops) >= 2 and correct:
                # Record successful composition
                comp_key = tuple(ops)
                self.pivot_lib.record_composition(comp_key, True)

        # Decay
        exp_p, exp_c = self.pivot_lib.decay(rate=0.005)

        return {
            'cycle': self.cycle,
            'feedback_processed': len(feedback_batch),
            'successes_in_batch': sum(1 for fb in feedback_batch if fb['correct']),
            'pivots_expired': exp_p,
            'compositions_expired': exp_c,
            'total_success_rate': self.total_successes / max(self.total_attempts, 1)
        }

    def get_true_state(self):
        """Ground truth state (for analysis only, NOT accessible to agent)"""
        all_prefs = np.zeros(UpgradedEnvironment.N_OPS)
        count = 0
        for p in self._internal_preferences.values():
            all_prefs += p
            count += 1
        if count > 0:
            all_prefs /= count

        return {
            'true_preferences': {i: round(float(p), 4) for i, p in enumerate(all_prefs)},
            'true_pivot_count': len(self.pivot_lib.pivots),
            'true_composition_count': len(self.pivot_lib.compositions),
            'op_graph_edges': self._internal_op_graph.number_of_edges(),
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'success_rate': round(self.total_successes / max(self.total_attempts, 1), 4),
            'energy_bits': self.energy_bits,
            'self_model_accuracy': round(1.0 - np.mean(np.abs(
                all_prefs - self.self_model['estimated_preferences']
            )), 4)
        }


# ============================================================
# EXPERIMENT
# ============================================================
def run_v02_experiment(n_episodes=600, report_every=60):
    print("=" * 70)
    print("TUF-E Grounded Agent v0.2: Lossy Self-Perception + Composable Pivots")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    env = UpgradedEnvironment(seed=42, feedback_delay=3)
    agent = GroundedAgentV02(probe_budget_per_cycle=3)

    results = {'milestones': [], 'final': {}, 'comparison': {}}
    pending_tasks = {}

    print(f"\nEnvironment: feedback_delay={env.feedback_delay}, partial observation, 8 ops")
    print(f"Agent: probe_budget=3/cycle, lossy self-model, composable pivots")

    for ep in range(n_episodes):
        difficulty = 1 if ep < n_episodes * 0.3 else (2 if ep < n_episodes * 0.6 else 3)
        task = env.generate_task(difficulty)
        pending_tasks[ep] = task

        observation = env.observe_partial(task['operands'])
        ops = agent.choose_actions(observation, len(task['operands']))
        feedback_batch = env.submit_attempt(task, ops, ep)

        if feedback_batch:
            tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in feedback_batch}
            learn_result = agent.learn_from_feedback_batch(feedback_batch, tasks_info)

            # Clean up old tasks
            for fb in feedback_batch:
                pending_tasks.pop(fb['episode'], None)

        # Periodic self-probe and report
        if (ep + 1) % report_every == 0:
            # Agent probes itself
            agent.probe_self(n_probes=3)

            true = agent.get_true_state()
            sm = agent.self_model

            top_true = sorted(true['true_preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
            top_self = sorted(enumerate(sm['estimated_preferences']), key=lambda x: x[1], reverse=True)[:3]

            OP_NAMES = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div', 4: 'neg', 5: 'abs', 6: 'sqr', 7: 'id'}

            true_names = [f"op_{k}({OP_NAMES[k]})={v:.3f}" for k, v in top_true]
            self_names = [f"op_{k}({OP_NAMES[k]})={v:.3f}" for k, v in top_self]

            print(f"  Ep {ep+1:>3}/{n_episodes}: "
                  f"SR={true['success_rate']:.3f} | "
                  f"pivots={true['true_pivot_count']} | "
                  f"comps={true['true_composition_count']} | "
                  f"self_acc={true['self_model_accuracy']:.3f} | "
                  f"conf={sm['confidence']:.2f}")
            print(f"    True top-3: {', '.join(true_names)}")
            print(f"    Self top-3: {', '.join(self_names)}")

            milestone = {
                'episode': ep + 1,
                'true_state': true,
                'self_model': {
                    'confidence': sm['confidence'],
                    'known_pivots': len(sm['known_pivots']),
                    'known_compositions': len(sm['known_compositions']),
                },
                'pivot_summary': agent.pivot_lib.get_summary()
            }
            results['milestones'].append(milestone)

    # Flush remaining feedback
    remaining = env.flush_feedback()
    if remaining:
        tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in remaining}
        agent.learn_from_feedback_batch(remaining, tasks_info)

    # Final state
    agent.probe_self(n_probes=3)
    final_true = agent.get_true_state()
    final_pivot_summary = agent.pivot_lib.get_summary()

    results['final'] = {
        'true_state': final_true,
        'pivot_summary': final_pivot_summary,
        'self_model': {
            'confidence': agent.self_model['confidence'],
            'known_pivots': agent.self_model['known_pivots'],
            'known_compositions': agent.self_model['known_compositions'],
        }
    }

    print(f"\n{'=' * 70}")
    print("Final Results")
    print(f"{'=' * 70}")
    print(f"  Success rate: {final_true['success_rate']:.3f} ({final_true['total_successes']}/{final_true['total_attempts']})")
    print(f"  Pivots learned: {final_true['true_pivot_count']}")
    print(f"  Compositions: {final_true['true_composition_count']}")
    print(f"  Self-model accuracy: {final_true['self_model_accuracy']:.3f}")
    print(f"  Energy: {final_true['energy_bits']:,} bits")
    print(f"  Op graph edges: {final_true['op_graph_edges']}")

    print(f"\n  Top pivots:")
    for p in final_pivot_summary.get('top_pivots', [])[:5]:
        print(f"    {p['pattern']}: strength={p['strength']}, successes={p['successes']}/{p['uses']}")

    print(f"\n  Top compositions:")
    OP_NAMES = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div', 4: 'neg', 5: 'abs', 6: 'sqr', 7: 'id'}
    for c in final_pivot_summary.get('top_compositions', [])[:5]:
        op_names = [OP_NAMES.get(o, '?') for o in c['ops']]
        print(f"    [{' → '.join(op_names)}]: strength={c['strength']}")

    # Compare self-model vs reality
    print(f"\n--- Self-Model vs Reality ---")
    print(f"  Self confidence: {agent.self_model['confidence']:.3f}")
    print(f"  Actual accuracy: {final_true['self_model_accuracy']:.3f}")
    print(f"  Self thinks it knows {len(agent.self_model['known_pivots'])} pivots, actually has {final_true['true_pivot_count']}")
    print(f"  Self thinks it knows {len(agent.self_model['known_compositions'])} compositions, actually has {final_true['true_composition_count']}")

    overestimate = len(agent.self_model['known_pivots']) > final_true['true_pivot_count']
    print(f"  Self-assessment: {'OVERCONFIDENT' if overestimate else 'CALIBRATED/UNDERCONFIDENT'}")

    results['comparison'] = {
        'self_confidence': agent.self_model['confidence'],
        'actual_accuracy': final_true['self_model_accuracy'],
        'self_pivot_estimate': len(agent.self_model['known_pivots']),
        'true_pivot_count': final_true['true_pivot_count'],
        'calibration': 'overconfident' if overestimate else 'calibrated'
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('grounding_v02_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_v02_results.json")

    return results


if __name__ == '__main__':
    run_v02_experiment(n_episodes=600, report_every=60)
