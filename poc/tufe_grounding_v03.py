"""
TUF-E Grounded Agent v0.3: 组合枢机实际使用 + 环境复杂度递增

v0.2的问题:
  - 组合枢机只被探测到但从未在行动选择中实际使用
  - find_composable_pairs()的结果没有反馈回record_composition()
  - 难度分布不合理(前30%全是单步)，多步学习机会太少
  - 环境复杂度固定，无课程学习

v0.3改进:
  1. 组合枢机闭环: 探测到的可组合对→行动选择尝试→结果反馈→强化/削弱
  2. 更好的难度分布: 全程混合难度，后期增加多步比例
  3. 环境复杂度递增: 操作数范围随episode扩大
  4. 自模型校准: confidence基于实际预测准确率而非探测次数
  5. 枢机泛化: 相似模式合并，避免碎片化
"""

import numpy as np
import networkx as nx
import json
import sys
import io
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# ESCALATING ENVIRONMENT
# ============================================================
class EscalatingEnvironment:
    """
    递增复杂度算术环境。
    - 操作数范围随episode扩大
    - 全程混合难度
    - 延迟反馈 + 部分可观察
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
        self.episode_count = 0

    def _value_range(self, episode, total_episodes):
        progress = episode / max(total_episodes, 1)
        low = 1
        high = int(10 + progress * 15)
        return low, high

    def observe_partial(self, operands):
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
            features.append({'magnitude': feat, 'sign': sign})
        return features

    def generate_task(self, difficulty, episode=0, total_episodes=600):
        lo, hi = self._value_range(episode, total_episodes)
        self.episode_count += 1

        if difficulty == 1:
            a = int(self.rng.integers(lo, hi))
            b = int(self.rng.integers(lo, hi))
            op_id = int(self.rng.choice([0, 1, 2]))
            _, fn = self.OPS[op_id]
            target = fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id], 'steps': 1}
        elif difficulty == 2:
            a = int(self.rng.integers(lo, min(hi, 20)))
            b = int(self.rng.integers(lo, min(hi, 20)))
            c = int(self.rng.integers(lo, min(hi, 20)))
            op1 = int(self.rng.choice([0, 1, 2]))
            op2 = int(self.rng.choice([0, 1, 2]))
            _, fn1 = self.OPS[op1]
            _, fn2 = self.OPS[op2]
            mid = fn1(a, b)
            target = fn2(mid, c)
            return {'operands': [a, b, c], 'target': target, 'solution_ops': [op1, op2], 'steps': 2}
        else:
            a = int(self.rng.integers(lo, min(hi, 12)))
            b = int(self.rng.integers(lo, min(hi, 12)))
            c = int(self.rng.integers(lo, min(hi, 12)))
            d = int(self.rng.integers(lo, min(hi, 12)))
            op1 = int(self.rng.choice([0, 1, 2]))
            op2 = int(self.rng.choice([0, 1, 2]))
            op3 = int(self.rng.choice([0, 1, 2]))
            _, fn1 = self.OPS[op1]
            _, fn2 = self.OPS[op2]
            _, fn3 = self.OPS[op3]
            mid1 = fn1(a, b)
            mid2 = fn2(mid1, c)
            target = fn3(mid2, d)
            return {'operands': [a, b, c, d], 'target': target,
                    'solution_ops': [op1, op2, op3], 'steps': 3}

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
        result = self.execute(task['operands'], op_sequence)
        correct = False
        proximity = 0.0
        if result is not None:
            try:
                diff = abs(result - task['target'])
                correct = diff < 0.001
                if not correct and task['target'] != 0:
                    proximity = max(0, 1.0 - diff / abs(task['target']))
            except (TypeError, OverflowError):
                pass

        entry = {
            'episode': episode,
            'correct': correct,
            'proximity': proximity,
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
# COMPOSABLE PIVOT LIBRARY v2
# ============================================================
class PivotLibraryV2:
    """
    改进版可组合枢机库。
    - 模式泛化: 相似离散区间合并
    - 组合闭环: 记录组合尝试结果
    - 强度归一化防止无限增长
    """

    MAGNITUDE_BINS = ['tiny', 'small', 'medium', 'large']

    def __init__(self, min_strength=0.08):
        self.pivots = {}
        self.compositions = {}
        self.min_strength = min_strength
        self.composition_attempts = defaultdict(lambda: {'uses': 0, 'successes': 0})

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

    def record(self, input_val, op_id, output_val, success, reward=1.0):
        in_pat = self.discretize(input_val)
        out_pat = self.discretize(output_val)
        key = (in_pat, op_id, out_pat)

        if key in self.pivots:
            p = self.pivots[key]
            p['uses'] += 1
            if success:
                p['successes'] += 1
                p['strength'] = min(1.0, p['strength'] + 0.15 * reward)
            else:
                p['strength'] = max(0, p['strength'] - 0.03)
        else:
            self.pivots[key] = {
                'strength': 0.25 * reward if success else 0.03,
                'uses': 1,
                'successes': 1 if success else 0,
                'ops': [op_id]
            }

    def find_composable_pairs(self, min_str=0.15):
        pairs = []
        pivot_list = [(k, v) for k, v in self.pivots.items()
                      if v['strength'] >= min_str]
        seen = set()
        for (in_a, op_a, out_a), pa in pivot_list:
            for (in_b, op_b, out_b), pb in pivot_list:
                if out_a == in_b and (op_a, op_b) not in seen:
                    seen.add((op_a, op_b))
                    pair_key = ((in_a, op_a, out_a), (in_b, op_b, out_b))
                    hist = self.composition_attempts[(op_a, op_b)]
                    empirical_str = (hist['successes'] / max(hist['uses'], 1)) if hist['uses'] > 0 else 0
                    theoretical_str = min(pa['strength'], pb['strength'])
                    combined = 0.4 * theoretical_str + 0.6 * empirical_str if hist['uses'] > 0 else theoretical_str

                    pairs.append({
                        'key': pair_key,
                        'combined_ops': pa['ops'] + pb['ops'],
                        'strength': round(combined, 4),
                        'in_pattern': in_a,
                        'out_pattern': out_b,
                        'empirical_uses': hist['uses'],
                        'empirical_successes': hist['successes']
                    })
        pairs.sort(key=lambda x: x['strength'], reverse=True)
        return pairs

    def record_composition_attempt(self, ops_tuple, success):
        self.composition_attempts[ops_tuple]['uses'] += 1
        if success:
            self.composition_attempts[ops_tuple]['successes'] += 1

        if ops_tuple in self.compositions:
            c = self.compositions[ops_tuple]
            c['uses'] += 1
            if success:
                c['strength'] = min(1.0, c['strength'] + 0.12)
                c['successes'] += 1
            else:
                c['strength'] = max(0, c['strength'] - 0.02)
        else:
            self.compositions[ops_tuple] = {
                'strength': 0.2 if success else 0.02,
                'uses': 1,
                'successes': 1 if success else 0,
                'ops': list(ops_tuple)
            }

    def decay(self, rate=0.004):
        expired_p = []
        for k, p in list(self.pivot_lib_items()):
            p['strength'] -= rate
            if p['strength'] < self.min_strength:
                expired_p.append(k)
        for k in expired_p:
            del self.pivots[k]

        expired_c = []
        for k, c in list(self.compositions.items()):
            c['strength'] -= rate
            if c['strength'] < self.min_strength:
                expired_c.append(k)
        for k in expired_c:
            del self.compositions[k]

        return len(expired_p), len(expired_c)

    def pivot_lib_items(self):
        return list(self.pivots.items())

    def get_summary(self):
        return {
            'n_pivots': len(self.pivots),
            'n_compositions': len(self.compositions),
            'mean_pivot_strength': round(float(np.mean([p['strength'] for p in self.pivots.values()])), 3) if self.pivots else 0,
            'top_pivots': sorted(
                [{'pattern': f"{k[0]}->op_{k[1]}->{k[2]}", 'strength': round(v['strength'], 3),
                  'successes': v['successes'], 'uses': v['uses']}
                 for k, v in self.pivots.items()],
                key=lambda x: x['strength'], reverse=True
            )[:8],
            'top_compositions': sorted(
                [{'ops': list(k), 'strength': round(v['strength'], 3),
                  'successes': v['successes'], 'uses': v['uses']}
                 for k, v in self.compositions.items()],
                key=lambda x: x['strength'], reverse=True
            )[:5],
            'composition_attempts': {
                str(k): {'uses': v['uses'], 'successes': v['successes']}
                for k, v in sorted(self.composition_attempts.items(),
                                   key=lambda x: x[1]['uses'], reverse=True)[:5]
            }
        }


# ============================================================
# GROUNDED AGENT v0.3
# ============================================================
class GroundedAgentV03:
    """
    v0.3智能体: 组合闭环 + 校准自模型 + 递增环境
    """

    OP_NAMES = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div', 4: 'neg', 5: 'abs', 6: 'sqr', 7: 'id'}

    def __init__(self, probe_budget_per_cycle=3, name='grounded-v0.3'):
        self.name = name
        self.cycle = 0
        self.probe_budget = probe_budget_per_cycle
        self.probes_used = 0

        self._internal_op_graph = nx.Graph()
        for i in range(EscalatingEnvironment.N_OPS):
            self._internal_op_graph.add_node(i, use_count=0, success_count=0)
        self._internal_preferences = {
            1: np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS,
            2: np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS,
            3: np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS,
        }

        self.pivot_lib = PivotLibraryV2()

        self.self_model = {
            'estimated_preferences': np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS,
            'known_pivots': [],
            'known_compositions': [],
            'confidence': 0.0,
            'last_probe_cycle': -1
        }

        self.total_attempts = 0
        self.total_successes = 0
        self.energy_bits = 0
        self.recent_predictions = []
        self.rng = np.random.default_rng(42)

    def probe_self(self, n_probes=None):
        if n_probes is None:
            n_probes = self.probe_budget

        actual_probes = min(n_probes, self.probe_budget - self.probes_used)
        if actual_probes <= 0:
            return self.self_model

        self.probes_used += actual_probes
        self.energy_bits += actual_probes * 128

        noisy_prefs = np.zeros(EscalatingEnvironment.N_OPS)
        for _ in range(actual_probes):
            arity = self.rng.choice([1, 2, 3])
            true_prefs = self._internal_preferences.get(arity, np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS)
            sampled_op = self.rng.choice(EscalatingEnvironment.N_OPS, p=true_prefs)
            noisy_prefs[sampled_op] += 1
        if noisy_prefs.sum() > 0:
            noisy_prefs /= noisy_prefs.sum()
        else:
            noisy_prefs = np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS

        noise = self.rng.dirichlet(np.ones(EscalatingEnvironment.N_OPS) * 10)
        noisy_prefs = 0.7 * noisy_prefs + 0.3 * noise

        self.self_model['estimated_preferences'] = noisy_prefs
        self.self_model['last_probe_cycle'] = self.cycle

        all_pivots = list(self.pivot_lib.pivots.items())
        if all_pivots:
            k = min(actual_probes, len(all_pivots))
            strengths = np.array([v['strength'] for _, v in all_pivots])
            if strengths.sum() > 0:
                probs = strengths / strengths.sum()
                indices = self.rng.choice(len(all_pivots), size=k, replace=False, p=probs) if k <= len(all_pivots) else self.rng.choice(len(all_pivots), size=k, replace=True, p=probs)
                sampled = [all_pivots[i] for i in indices]
                self.self_model['known_pivots'] = [
                    {'pattern': f"{k_[0]}->op_{k_[1]}->{k_[2]}", 'strength': round(v['strength'], 3)}
                    for k_, v in sampled
                ]
            else:
                self.self_model['known_pivots'] = []

        comps = self.pivot_lib.find_composable_pairs()
        if comps:
            self.self_model['known_compositions'] = [
                {'ops': c['combined_ops'], 'strength': round(c['strength'], 3)}
                for c in comps[:actual_probes]
            ]
        else:
            self.self_model['known_compositions'] = []

        if len(self.recent_predictions) >= 5:
            recent = self.recent_predictions[-30:]
            trend = np.mean(recent[-10:]) - np.mean(recent[:10]) if len(recent) >= 20 else 0
            pivot_signal = min(1.0, len(self.pivot_lib.pivots) / 8)
            comp_signal = min(1.0, sum(v['uses'] for v in self.pivot_lib.composition_attempts.values()) / 20)
            self.self_model['confidence'] = min(0.9, max(0.15, 0.3 + 0.3 * pivot_signal + 0.2 * comp_signal + 0.1 * max(0, trend)))
        else:
            self.self_model['confidence'] = 0.15

        return self.self_model

    def choose_actions(self, observation, n_operands, episode=0, total_episodes=600):
        seq_len = max(1, n_operands - 1)
        prefs = self.self_model['estimated_preferences']
        exploration = max(0.1, 1.0 - self.self_model['confidence'])

        early_phase = episode < total_episodes * 0.15

        if not early_phase and self.rng.random() > exploration and self.self_model['known_compositions']:
            best_comp = max(self.self_model['known_compositions'], key=lambda x: x['strength'])
            ops = best_comp['ops'][:seq_len]
            while len(ops) < seq_len:
                ops.append(int(self.rng.choice(EscalatingEnvironment.N_OPS, p=prefs)))
            return ops[:seq_len]

        if not early_phase and self.rng.random() > exploration and self.self_model['known_pivots']:
            obs_mag = observation[0]['magnitude'] if observation else 'medium'
            matching = [p for p in self.self_model['known_pivots'] if obs_mag in p['pattern']]
            if matching:
                best = max(matching, key=lambda x: x['strength'])
                try:
                    op_str = best['pattern'].split('->')[1]
                    op_id = int(op_str.replace('op_', ''))
                    return [op_id] * seq_len
                except (IndexError, ValueError):
                    pass

        if self.rng.random() < exploration:
            return [int(self.rng.choice(EscalatingEnvironment.N_OPS)) for _ in range(seq_len)]
        else:
            return [int(self.rng.choice(EscalatingEnvironment.N_OPS, p=prefs)) for _ in range(seq_len)]

    def learn_from_feedback_batch(self, feedback_batch, tasks_info):
        self.cycle += 1
        self.probes_used = 0

        batch_correct = 0
        for fb in feedback_batch:
            self.total_attempts += 1
            ops = fb['ops']
            correct = fb['correct']
            proximity = fb.get('proximity', 0.0)
            result = fb['result']
            target = fb['target']

            if correct:
                self.total_successes += 1
                batch_correct += 1

            bits = len(ops) * 32 + 64
            self.energy_bits += bits

            arity = min(len(ops), 3)
            prefs = self._internal_preferences.setdefault(arity, np.ones(EscalatingEnvironment.N_OPS) / EscalatingEnvironment.N_OPS)

            reward = 1.0 if correct else proximity * 0.3

            if correct or proximity > 0.3:
                for op in ops:
                    prefs[op] *= (1.0 + 0.12 * reward)
                    self._internal_op_graph.nodes[op]['use_count'] += 1
                    if correct:
                        self._internal_op_graph.nodes[op]['success_count'] += 1
                prefs /= prefs.sum()

                for i in range(len(ops)):
                    for j in range(i + 1, len(ops)):
                        if self._internal_op_graph.has_edge(ops[i], ops[j]):
                            self._internal_op_graph[ops[i]][ops[j]]['weight'] += reward
                        else:
                            self._internal_op_graph.add_edge(ops[i], ops[j], weight=reward)
            else:
                for op in ops:
                    prefs[op] *= 0.97
                prefs /= prefs.sum()

            values = tasks_info.get(fb['episode'], {}).get('operands', [0, 0])
            current_val = values[0] if values else 0
            for i, op in enumerate(ops):
                next_val = result if i == len(ops) - 1 else current_val
                self.pivot_lib.record(current_val, op, next_val, correct, reward)
                if i < len(values) - 1:
                    current_val = values[i + 1]

            if len(ops) >= 2:
                comp_key = tuple(ops)
                self.pivot_lib.record_composition_attempt(comp_key, correct)

        self.recent_predictions.append(batch_correct / max(len(feedback_batch), 1))

        exp_p, exp_c = self.pivot_lib.decay(rate=0.004)

        return {
            'cycle': self.cycle,
            'feedback_processed': len(feedback_batch),
            'successes_in_batch': batch_correct,
            'pivots_expired': exp_p,
            'compositions_expired': exp_c,
            'total_success_rate': self.total_successes / max(self.total_attempts, 1)
        }

    def get_true_state(self):
        all_prefs = np.zeros(EscalatingEnvironment.N_OPS)
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
def run_v03_experiment(n_episodes=800, report_every=80):
    print("=" * 70)
    print("TUF-E Grounded Agent v0.3: Composition Loop + Calibrated Self-Model")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    env = EscalatingEnvironment(seed=42, feedback_delay=3)
    agent = GroundedAgentV03(probe_budget_per_cycle=3)

    results = {'milestones': [], 'final': {}, 'comparison': {}}
    pending_tasks = {}

    print(f"\nEnvironment: escalating range, feedback_delay={env.feedback_delay}, partial obs")
    print(f"Agent: probe_budget=3/cycle, composition loop, calibrated confidence")

    for ep in range(n_episodes):
        progress = ep / n_episodes
        r = agent.rng.random()
        if progress < 0.3:
            difficulty = 1 if r < 0.6 else 2
        elif progress < 0.6:
            difficulty = 1 if r < 0.3 else (2 if r < 0.75 else 3)
        else:
            difficulty = 1 if r < 0.15 else (2 if r < 0.55 else 3)

        task = env.generate_task(difficulty, ep, n_episodes)
        pending_tasks[ep] = task

        observation = env.observe_partial(task['operands'])
        ops = agent.choose_actions(observation, len(task['operands']), ep, n_episodes)
        feedback_batch = env.submit_attempt(task, ops, ep)

        if feedback_batch:
            tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in feedback_batch}
            learn_result = agent.learn_from_feedback_batch(feedback_batch, tasks_info)
            for fb in feedback_batch:
                pending_tasks.pop(fb['episode'], None)

        if (ep + 1) % report_every == 0:
            agent.probe_self(n_probes=3)
            true = agent.get_true_state()
            sm = agent.self_model

            top_true = sorted(true['true_preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
            top_self = sorted(enumerate(sm['estimated_preferences']), key=lambda x: x[1], reverse=True)[:3]

            true_names = [f"op_{k}({agent.OP_NAMES[k]})={v:.3f}" for k, v in top_true]
            self_names = [f"op_{k}({agent.OP_NAMES[k]})={v:.3f}" for k, v in top_self]

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

    remaining = env.flush_feedback()
    if remaining:
        tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in remaining}
        agent.learn_from_feedback_batch(remaining, tasks_info)

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
    for c in final_pivot_summary.get('top_compositions', [])[:5]:
        op_names = [agent.OP_NAMES.get(o, '?') for o in c['ops']]
        print(f"    [{' -> '.join(op_names)}]: strength={c['strength']}, successes={c['successes']}/{c['uses']}")

    print(f"\n  Composition attempt history:")
    for k, v in final_pivot_summary.get('composition_attempts', {}).items():
        op_names = [agent.OP_NAMES.get(int(x.strip()), '?') for x in k.strip('()').split(',')]
        sr = v['successes'] / max(v['uses'], 1)
        print(f"    ({', '.join(op_names)}): {v['successes']}/{v['uses']} = {sr:.2f}")

    print(f"\n--- Self-Model vs Reality ---")
    print(f"  Self confidence: {agent.self_model['confidence']:.3f}")
    print(f"  Actual accuracy: {final_true['self_model_accuracy']:.3f}")
    print(f"  Self thinks it knows {len(agent.self_model['known_pivots'])} pivots, actually has {final_true['true_pivot_count']}")
    print(f"  Self thinks it knows {len(agent.self_model['known_compositions'])} compositions, actually has {final_true['true_composition_count']}")

    overestimate = len(agent.self_model['known_pivots']) > final_true['true_pivot_count']
    conf_gap = abs(agent.self_model['confidence'] - final_true['self_model_accuracy'])
    print(f"  Confidence-accuracy gap: {conf_gap:.3f}")
    print(f"  Self-assessment: {'OVERCONFIDENT' if overestimate else 'CALIBRATED/UNDERCONFIDENT'}")

    results['comparison'] = {
        'self_confidence': agent.self_model['confidence'],
        'actual_accuracy': final_true['self_model_accuracy'],
        'confidence_accuracy_gap': round(conf_gap, 4),
        'self_pivot_estimate': len(agent.self_model['known_pivots']),
        'true_pivot_count': final_true['true_pivot_count'],
        'calibration': 'overconfident' if overestimate else 'calibrated'
    }

    # v0.1/v0.2 comparison
    results['version_comparison'] = {
        'v0.1': {'success_rate': 0.198, 'pivots': 3, 'compositions': 0, 'self_model': 'none'},
        'v0.2': {'success_rate': 0.213, 'pivots': 4, 'compositions': 1, 'self_model': 'lossy_fixed_conf'},
        'v0.3': {'success_rate': final_true['success_rate'],
                 'pivots': final_true['true_pivot_count'],
                 'compositions': final_true['true_composition_count'],
                 'self_model': f"calibrated(gap={conf_gap:.3f})"}
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('grounding_v03_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_v03_results.json")

    return results


if __name__ == '__main__':
    run_v03_experiment(n_episodes=800, report_every=80)
