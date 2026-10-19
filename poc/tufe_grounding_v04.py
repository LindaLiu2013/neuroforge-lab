"""
TUF-E Grounded Agent v0.4: 多智能体异构共识

v0.3的发现:
  - 组合枢机闭环可行，但单智能体受限于自身感知偏差
  - add主导(29%)而非mul，因为递增范围下add更稳定
  - 自模型校准改善(conf-acc gap=0.14)，但仍有过高/过低估计
  - 核心瓶颈: 单一观察者的感知构造器限制了可发现的枢机空间

v0.4设计:
  1. 3个异构智能体，各有不同的感知偏差和初始偏好
     - Agent-A: 偏向数值大小(magnitude-sensitive)
     - Agent-B: 偏向符号特征(sign/parity-sensitive)
     - Agent-C: 偏向操作历史(history-sensitive)
  2. C3共识协议: 只有≥2个智能体独立发现相同枢机时才入库
     - 模拟Tier 2 Module D的弱连接共识机制
     - 实现Q0.5异构观察者交叉验证
  3. 共享知识库 + 私有感知通道
     - 知识是公共的(耗散结构)
     - 感知是私有的(构造器不可还原)
  4. 对比实验: 共识 vs 无共识 vs 单智能体
"""

import numpy as np
import networkx as nx
import json
import sys
import io
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# SHARED ENVIRONMENT (same as v0.3 but stable range)
# ============================================================
class SharedEnvironment:
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

    def generate_task(self, difficulty=1):
        if difficulty == 1:
            a = int(self.rng.integers(1, 20))
            b = int(self.rng.integers(1, 20))
            op_id = int(self.rng.choice([0, 1, 2]))
            _, fn = self.OPS[op_id]
            target = fn(a, b)
            return {'operands': [a, b], 'target': target, 'solution_ops': [op_id], 'steps': 1}
        elif difficulty == 2:
            a = int(self.rng.integers(1, 12))
            b = int(self.rng.integers(1, 12))
            c = int(self.rng.integers(1, 12))
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
# HETEROGENEOUS OBSERVER
# ============================================================
class HeterogeneousObserver:
    """
    每个观察者有不同的感知构造器。
    同一输入，不同观察者看到不同的特征空间。
    """

    MAGNITUDE_BINS = ['tiny', 'small', 'medium', 'large']

    def __init__(self, observer_type, rng):
        self.observer_type = observer_type
        self.rng = rng

    def perceive(self, operands):
        if self.observer_type == 'magnitude':
            return self._perceive_magnitude(operands)
        elif self.observer_type == 'symbolic':
            return self._perceive_symbolic(operands)
        elif self.observer_type == 'temporal':
            return self._perceive_temporal(operands)
        return self._perceive_magnitude(operands)

    def _perceive_magnitude(self, operands):
        features = []
        for v in operands:
            av = abs(v)
            if av < 5: mag = 'tiny'
            elif av < 15: mag = 'small'
            elif av < 50: mag = 'medium'
            else: mag = 'large'
            features.append({'magnitude': mag, 'sign': 'pos' if v >= 0 else 'neg'})
        return features

    def _perceive_symbolic(self, operands):
        features = []
        for v in operands:
            parity = 'even' if v % 2 == 0 else 'odd'
            sign = 'pos' if v >= 0 else 'neg'
            prime_like = 'simple' if abs(v) in {1, 2, 3, 5, 7, 11, 13} else 'complex'
            features.append({'parity': parity, 'sign': sign, 'complexity': prime_like})
        return features

    def _perceive_temporal(self, operands):
        features = []
        for v in operands:
            av = abs(v)
            if av < 5: mag = 'tiny'
            elif av < 15: mag = 'small'
            elif av < 50: mag = 'medium'
            else: mag = 'large'
            noise = self.rng.choice(['clear', 'noisy'])
            features.append({'magnitude': mag, 'clarity': noise})
        return features

    def discretize_for_pivot(self, value):
        if self.observer_type == 'magnitude':
            av = abs(value) if value is not None else 0
            if av < 5: mag = 'tiny'
            elif av < 15: mag = 'small'
            elif av < 50: mag = 'medium'
            else: mag = 'large'
            sign = 'pos' if (value or 0) >= 0 else 'neg'
            return f"{mag}_{sign}"
        elif self.observer_type == 'symbolic':
            v = value or 0
            parity = 'even' if v % 2 == 0 else 'odd'
            sign = 'pos' if v >= 0 else 'neg'
            return f"{parity}_{sign}"
        elif self.observer_type == 'temporal':
            av = abs(value) if value is not None else 0
            if av < 10: mag = 'lo'
            elif av < 30: mag = 'mid'
            else: mag = 'hi'
            return mag
        return str(value)


# ============================================================
# CONSENSUS KNOWLEDGE STORE
# ============================================================
class ConsensusKnowledgeStore:
    """
    C3共识知识库: 只有≥min_observers个独立观察者确认的枢机才入库。

    v0.4关键修正: 不同观察者的私有离散化空间不可直接比较。
    需要一个共享抽象层(shared abstraction)作为翻译协议。
    提案格式统一为 (op_id, input_range_bucket, output_range_bucket)，
    这是所有观察者都能映射到的最小公共语义空间。
    """

    SHARED_BUCKETS = ['tiny', 'small', 'medium', 'large']

    def __init__(self, min_observers=2, min_strength=0.15):
        self.candidates = {}
        self.confirmed = {}
        self.min_observers = min_observers
        self.min_strength = min_strength

    @staticmethod
    def to_shared_key(pivot_key, observer_type):
        """将私有pivot key转换为共享抽象空间的key"""
        if not isinstance(pivot_key, tuple) or len(pivot_key) < 3:
            return None
        in_pat, op_id, out_pat = pivot_key[0], pivot_key[1], pivot_key[2]

        if observer_type == 'magnitude':
            in_bucket = in_pat.split('_')[0] if '_' in str(in_pat) else 'medium'
            out_bucket = out_pat.split('_')[0] if '_' in str(out_pat) else 'medium'
        elif observer_type == 'symbolic':
            in_bucket = 'medium'
            out_bucket = 'medium'
            if 'even' in str(in_pat): in_bucket = 'small'
            if 'odd' in str(in_pat): in_bucket = 'medium'
            if 'even' in str(out_pat): out_bucket = 'small'
            if 'odd' in str(out_pat): out_bucket = 'medium'
        elif observer_type == 'temporal':
            mapping = {'lo': 'tiny', 'mid': 'small', 'hi': 'large'}
            in_bucket = mapping.get(str(in_pat), 'medium')
            out_bucket = mapping.get(str(out_pat), 'medium')
        else:
            in_bucket = 'medium'
            out_bucket = 'medium'

        return (in_bucket, op_id, out_bucket)

    def propose(self, pivot_key, observer_id, strength, ops, observer_type='magnitude'):
        shared_key = self.to_shared_key(pivot_key, observer_type)
        if shared_key is None:
            return False

        if shared_key not in self.candidates:
            self.candidates[shared_key] = {
                'proposals': {},
                'ops': ops,
                'total_strength': 0,
                'n_proposers': 0
            }
        c = self.candidates[shared_key]
        if observer_id not in c['proposals']:
            c['proposals'][observer_id] = strength
            c['n_proposers'] += 1
            c['total_strength'] += strength

        if c['n_proposers'] >= self.min_observers:
            avg_str = c['total_strength'] / c['n_proposers']
            if avg_str >= self.min_strength:
                self.confirmed[shared_key] = {
                    'strength': avg_str,
                    'n_confirmers': c['n_proposers'],
                    'ops': ops,
                    'confirmers': list(c['proposals'].keys())
                }
                return True
        return False

    def reinforce(self, pivot_key, success):
        if pivot_key in self.confirmed:
            c = self.confirmed[pivot_key]
            if success:
                c['strength'] = min(1.0, c['strength'] + 0.1)
            else:
                c['strength'] = max(0, c['strength'] - 0.03)
                if c['strength'] < self.min_strength:
                    del self.confirmed[pivot_key]

    def decay(self, rate=0.003):
        expired = []
        for k, c in list(self.confirmed.items()):
            c['strength'] -= rate
            if c['strength'] < self.min_strength:
                expired.append(k)
        for k in expired:
            del self.confirmed[k]
        return len(expired)

    def get_confirmed_pivots(self):
        return dict(self.confirmed)

    def get_summary(self):
        return {
            'n_candidates': len(self.candidates),
            'n_confirmed': len(self.confirmed),
            'top_confirmed': sorted(
                [{'key': str(k), 'strength': round(v['strength'], 3),
                  'n_confirmers': v['n_confirmers'], 'ops': v['ops']}
                 for k, v in self.confirmed.items()],
                key=lambda x: x['strength'], reverse=True
            )[:8]
        }


# ============================================================
# INDIVIDUAL AGENT (within multi-agent system)
# ============================================================
class IndividualAgent:
    OP_NAMES = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div', 4: 'neg', 5: 'abs', 6: 'sqr', 7: 'id'}

    def __init__(self, agent_id, observer_type, initial_bias=None, seed=42):
        self.agent_id = agent_id
        self.observer = HeterogeneousObserver(observer_type, np.random.default_rng(seed))
        self.rng = np.random.default_rng(seed + hash(agent_id) % 10000)

        self.preferences = np.ones(SharedEnvironment.N_OPS) / SharedEnvironment.N_OPS
        if initial_bias:
            for op, weight in initial_bias.items():
                self.preferences[op] *= weight
            self.preferences /= self.preferences.sum()

        self.private_pivots = {}
        self.total_attempts = 0
        self.total_successes = 0
        self.energy_bits = 0

    def perceive(self, operands):
        return self.observer.perceive(operands)

    def choose_action(self, observation, n_operands, confirmed_pivots):
        seq_len = max(1, n_operands - 1)

        if confirmed_pivots and self.rng.random() < 0.4:
            obs_features = observation[0] if observation else {}
            matching = []
            for pk, pv in confirmed_pivots.items():
                if isinstance(pk, tuple) and len(pk) >= 3:
                    in_pat = pk[0]
                    for feat_val in obs_features.values():
                        if feat_val in str(in_pat):
                            matching.append((pk, pv))
                            break
            if matching:
                best = max(matching, key=lambda x: x[1]['strength'])
                op_id = best[0][1] if isinstance(best[0][1], int) else self.rng.choice(SharedEnvironment.N_OPS, p=self.preferences)
                return [op_id] * seq_len

        if self.rng.random() < 0.15:
            return [int(self.rng.choice(SharedEnvironment.N_OPS)) for _ in range(seq_len)]
        return [int(self.rng.choice(SharedEnvironment.N_OPS, p=self.preferences)) for _ in range(seq_len)]

    def learn(self, feedback_batch, tasks_info, consensus_store):
        batch_correct = 0
        proposed_pivots = []

        for fb in feedback_batch:
            self.total_attempts += 1
            ops = fb['ops']
            correct = fb['correct']
            proximity = fb.get('proximity', 0.0)
            result = fb['result']

            if correct:
                self.total_successes += 1
                batch_correct += 1

            self.energy_bits += len(ops) * 32 + 64

            reward = 1.0 if correct else proximity * 0.3
            if correct or proximity > 0.3:
                for op in ops:
                    self.preferences[op] *= (1.0 + 0.12 * reward)
                self.preferences /= self.preferences.sum()
            else:
                for op in ops:
                    self.preferences[op] *= 0.97
                self.preferences /= self.preferences.sum()

            values = tasks_info.get(fb['episode'], {}).get('operands', [0, 0])
            current_val = values[0] if values else 0
            for i, op in enumerate(ops):
                next_val = result if i == len(ops) - 1 else current_val
                in_pat = self.observer.discretize_for_pivot(current_val)
                out_pat = self.observer.discretize_for_pivot(next_val)
                pivot_key = (in_pat, op, out_pat)

                if correct:
                    if pivot_key in self.private_pivots:
                        self.private_pivots[pivot_key]['strength'] = min(1.0, self.private_pivots[pivot_key]['strength'] + 0.15)
                        self.private_pivots[pivot_key]['successes'] += 1
                    else:
                        self.private_pivots[pivot_key] = {'strength': 0.25, 'successes': 1, 'uses': 1}

                    str_val = self.private_pivots[pivot_key]['strength']
                    consensus_store.propose(pivot_key, self.agent_id, str_val, [op], self.observer.observer_type)
                    proposed_pivots.append(pivot_key)

                if i < len(values) - 1:
                    current_val = values[i + 1]

        for pk in proposed_pivots:
            if pk in consensus_store.confirmed:
                self.private_pivots.setdefault(pk, {'strength': 0.1, 'successes': 0, 'uses': 0})
                self.private_pivots[pk]['strength'] = min(1.0, self.private_pivots[pk]['strength'] + 0.05)

        return batch_correct

    def get_stats(self):
        return {
            'agent_id': self.agent_id,
            'observer_type': self.observer.observer_type,
            'total_attempts': self.total_attempts,
            'total_successes': self.total_successes,
            'success_rate': round(self.total_successes / max(self.total_attempts, 1), 4),
            'n_private_pivots': len(self.private_pivots),
            'top_preferences': {i: round(float(p), 4) for i, p in enumerate(self.preferences)},
            'energy_bits': self.energy_bits
        }


# ============================================================
# MULTI-AGENT SYSTEM
# ============================================================
class MultiAgentSystem:
    def __init__(self, use_consensus=True, seed=42):
        self.use_consensus = use_consensus
        self.consensus_store = ConsensusKnowledgeStore(min_observers=2)

        self.agents = [
            IndividualAgent('A', 'magnitude',
                          initial_bias={0: 1.5, 2: 1.3}, seed=seed),
            IndividualAgent('B', 'symbolic',
                          initial_bias={1: 1.5, 3: 1.3}, seed=seed + 1),
            IndividualAgent('C', 'temporal',
                          initial_bias={2: 1.5, 6: 1.3}, seed=seed + 2),
        ]

    def run_episode(self, env, task, episode, pending_tasks):
        observation_base = task['operands']
        chosen_agent_idx = int(np.random.default_rng(episode).choice(len(self.agents)))
        agent = self.agents[chosen_agent_idx]

        observation = agent.perceive(observation_base)
        confirmed = self.consensus_store.get_confirmed_pivots() if self.use_consensus else {}
        ops = agent.choose_action(observation, len(task['operands']), confirmed)

        return env.submit_attempt(task, ops, episode), chosen_agent_idx

    def process_feedback(self, feedback_batch, pending_tasks):
        if not feedback_batch:
            return None

        tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in feedback_batch}

        total_correct = 0
        for agent in self.agents:
            bc = agent.learn(feedback_batch, tasks_info, self.consensus_store)
            total_correct += bc

        if self.use_consensus:
            self.consensus_store.decay(rate=0.003)

        return {
            'feedback_processed': len(feedback_batch),
            'total_correct_across_agents': total_correct,
            'confirmed_pivots': len(self.consensus_store.confirmed) if self.use_consensus else 0
        }

    def get_system_stats(self):
        stats = {
            'agent_stats': [a.get_stats() for a in self.agents],
            'consensus': self.consensus_store.get_summary() if self.use_consensus else {'n_confirmed': 0},
        }

        total_att = sum(a.total_attempts for a in self.agents)
        total_suc = sum(a.total_successes for a in self.agents)
        stats['system_success_rate'] = round(total_suc / max(total_att, 1), 4)
        stats['total_energy'] = sum(a.energy_bits for a in self.agents)
        return stats


# ============================================================
# SINGLE AGENT BASELINE (for comparison)
# ============================================================
def run_single_agent_baseline(n_episodes=600, seed=42):
    env = SharedEnvironment(seed=seed, feedback_delay=3)
    agent = IndividualAgent('solo', 'magnitude', seed=seed)
    dummy_store = ConsensusKnowledgeStore(min_observers=99)
    pending_tasks = {}
    successes = 0
    attempts = 0

    for ep in range(n_episodes):
        r = np.random.default_rng(ep).random()
        difficulty = 1 if r < 0.5 else (2 if r < 0.8 else 3)
        task = env.generate_task(difficulty)
        pending_tasks[ep] = task

        obs = agent.perceive(task['operands'])
        ops = agent.choose_action(obs, len(task['operands']), {})
        fb_batch = env.submit_attempt(task, ops, ep)

        if fb_batch:
            tasks_info = {fb['episode']: pending_tasks.get(fb['episode'], {}) for fb in fb_batch}
            for fb in fb_batch:
                attempts += 1
                if fb['correct']:
                    successes += 1
            agent.learn(fb_batch, tasks_info, dummy_store)
            for fb in fb_batch:
                pending_tasks.pop(fb['episode'], None)

    remaining = env.flush_feedback()
    if remaining:
        for fb in remaining:
            attempts += 1
            if fb['correct']:
                successes += 1

    return {
        'success_rate': round(successes / max(attempts, 1), 4),
        'attempts': attempts,
        'successes': successes,
        'n_private_pivots': len(agent.private_pivots),
        'energy_bits': agent.energy_bits
    }


# ============================================================
# EXPERIMENT
# ============================================================
def run_v04_experiment(n_episodes=600, report_every=60):
    print("=" * 70)
    print("TUF-E Grounded Agent v0.4: Multi-Agent Heterogeneous Consensus")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {'milestones': [], 'final': {}, 'comparison': {}}

    # --- Run WITH consensus ---
    print("\n--- Condition A: Multi-Agent WITH Consensus (C3) ---")
    env_a = SharedEnvironment(seed=42, feedback_delay=3)
    system_a = MultiAgentSystem(use_consensus=True, seed=42)
    pending_a = {}

    for ep in range(n_episodes):
        r = np.random.default_rng(ep).random()
        difficulty = 1 if r < 0.5 else (2 if r < 0.8 else 3)
        task = env_a.generate_task(difficulty)
        pending_a[ep] = task

        fb_batch, agent_idx = system_a.run_episode(env_a, task, ep, pending_a)
        if fb_batch:
            learn_result = system_a.process_feedback(fb_batch, pending_a)
            for fb in fb_batch:
                pending_a.pop(fb['episode'], None)

        if (ep + 1) % report_every == 0:
            stats = system_a.get_system_stats()
            cs = stats['consensus']
            agent_srs = ', '.join(f"{a['success_rate']:.2f}" for a in stats['agent_stats'])
            print(f"  Ep {ep+1:>3}/{n_episodes}: "
                  f"SR={stats['system_success_rate']:.3f} | "
                  f"confirmed={cs['n_confirmed']} | "
                  f"candidates={cs['n_candidates']} | "
                  f"agents=[{agent_srs}]")

            results['milestones'].append({
                'episode': ep + 1,
                'condition': 'consensus',
                'system_sr': stats['system_success_rate'],
                'confirmed_pivots': cs['n_confirmed'],
                'agent_srs': [a['success_rate'] for a in stats['agent_stats']]
            })

    remaining_a = env_a.flush_feedback()
    if remaining_a:
        system_a.process_feedback(remaining_a, pending_a)

    final_a = system_a.get_system_stats()
    results['final']['consensus'] = final_a

    print(f"\n  Final (Consensus): SR={final_a['system_success_rate']:.3f}")
    print(f"  Confirmed pivots: {final_a['consensus']['n_confirmed']}")
    print(f"  Candidates evaluated: {final_a['consensus']['n_candidates']}")
    for a in final_a['agent_stats']:
        print(f"    Agent-{a['agent_id']} ({a['observer_type']}): SR={a['success_rate']:.3f}, pivots={a['n_private_pivots']}")

    if final_a['consensus'].get('top_confirmed'):
        print(f"  Top confirmed pivots:")
        for p in final_a['consensus']['top_confirmed'][:5]:
            print(f"    {p['key']}: str={p['strength']}, confirmers={p['n_confirmers']}")

    # --- Run WITHOUT consensus ---
    print(f"\n--- Condition B: Multi-Agent WITHOUT Consensus ---")
    env_b = SharedEnvironment(seed=42, feedback_delay=3)
    system_b = MultiAgentSystem(use_consensus=False, seed=42)
    pending_b = {}

    for ep in range(n_episodes):
        r = np.random.default_rng(ep).random()
        difficulty = 1 if r < 0.5 else (2 if r < 0.8 else 3)
        task = env_b.generate_task(difficulty)
        pending_b[ep] = task

        fb_batch, agent_idx = system_b.run_episode(env_b, task, ep, pending_b)
        if fb_batch:
            system_b.process_feedback(fb_batch, pending_b)
            for fb in fb_batch:
                pending_b.pop(fb['episode'], None)

    remaining_b = env_b.flush_feedback()
    if remaining_b:
        system_b.process_feedback(remaining_b, pending_b)

    final_b = system_b.get_system_stats()
    results['final']['no_consensus'] = final_b

    print(f"  Final (No Consensus): SR={final_b['system_success_rate']:.3f}")
    for a in final_b['agent_stats']:
        print(f"    Agent-{a['agent_id']} ({a['observer_type']}): SR={a['success_rate']:.3f}, pivots={a['n_private_pivots']}")

    # --- Single agent baseline ---
    print(f"\n--- Condition C: Single Agent Baseline ---")
    solo = run_single_agent_baseline(n_episodes=n_episodes, seed=42)
    results['final']['single_agent'] = solo
    print(f"  Final (Single): SR={solo['success_rate']:.3f}, pivots={solo['n_private_pivots']}")

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")

    sr_consensus = final_a['system_success_rate']
    sr_no_consensus = final_b['system_success_rate']
    sr_single = solo['success_rate']

    print(f"  Multi-Agent + Consensus:  SR={sr_consensus:.3f}, confirmed_pivots={final_a['consensus']['n_confirmed']}")
    print(f"  Multi-Agent No Consensus: SR={sr_no_consensus:.3f}")
    print(f"  Single Agent Baseline:    SR={sr_single:.3f}, private_pivots={solo['n_private_pivots']}")

    consensus_boost = sr_consensus - sr_single
    no_consensus_boost = sr_no_consensus - sr_single
    consensus_vs_no = sr_consensus - sr_no_consensus

    print(f"\n  Consensus boost over single:   {consensus_boost:+.3f}")
    print(f"  No-consensus boost over single: {no_consensus_boost:+.3f}")
    print(f"  Consensus vs no-consensus:      {consensus_vs_no:+.3f}")

    if consensus_boost > 0.01:
        verdict = "CONSENSUS HELPS"
    elif consensus_vs_no > 0.01:
        verdict = "MULTI-AGENT HELPS, CONSENSUS MARGINAL"
    else:
        verdict = "NO CLEAR BENEFIT FROM CONSENSUS"

    print(f"\n  Verdict: {verdict}")

    results['comparison'] = {
        'consensus_sr': sr_consensus,
        'no_consensus_sr': sr_no_consensus,
        'single_sr': sr_single,
        'consensus_boost': round(consensus_boost, 4),
        'consensus_vs_no': round(consensus_vs_no, 4),
        'verdict': verdict,
        'confirmed_pivots': final_a['consensus']['n_confirmed'],
        'energy_consensus': final_a['total_energy'],
        'energy_single': solo['energy_bits']
    }

    # Version history
    results['version_comparison'] = {
        'v0.1': {'sr': 0.198, 'type': 'single', 'consensus': False},
        'v0.2': {'sr': 0.213, 'type': 'single+lossy_self', 'consensus': False},
        'v0.3': {'sr': 0.131, 'type': 'single+composition', 'consensus': False},
        'v0.4_consensus': {'sr': sr_consensus, 'type': 'multi-agent+consensus', 'consensus': True},
        'v0.4_no_consensus': {'sr': sr_no_consensus, 'type': 'multi-agent', 'consensus': False},
        'v0.4_single': {'sr': sr_single, 'type': 'single_baseline', 'consensus': False},
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('grounding_v04_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_v04_results.json")

    return results


if __name__ == '__main__':
    run_v04_experiment(n_episodes=600, report_every=60)
