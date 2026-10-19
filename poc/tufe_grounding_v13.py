"""
TUF-E v1.3: 食堂管理域 + 完整枢机机制迁移

将v1.2验证过的枢机机制(16桶/三观察者/C3共识/Betti检测/RAF闭合/四层验证)
迁移到v0.5的CafeteriaEnvironment。

核心适配:
- 三个异构观察者从数值空间改为食堂观测空间:
  * structural: 状态快照(库存/人员/设备的离散化)
  * temporal: 趋势变化(连续两天的差异模式)
  * functional: 因果链完整性(采购→储存→加工→洗消的覆盖度)
- 枢机三元组: (situation_pattern, action_type, outcome_pattern)
- 保留CafeteriaEnvironment不变(Jayson验证过的真实域模型)

[DRIFT-CHECK] 不使用reward/maximize等变量名
[DRIFT-CHECK] 学习信号 = "此行为是否维持可行空间?"
[ME-C4] 所有AI推断标注置信度
"""

import numpy as np
import networkx as nx
import json
import sys
import io
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

# stdout encoding handled by imported v0.5 module

# 导入v0.5环境(不修改)
sys.path.insert(0, '.')
from tufe_grounding_v05 import CafeteriaEnvironment


# ============================================================
# 食堂域异构观察者
# ============================================================
class CafeteriaObserver:
    """
    食堂域的异构观察者。
    三种视角对同一观测产生不同的离散化模式。
    """

    def __init__(self, observer_type, seed=42):
        self.observer_type = observer_type
        self.rng = np.random.default_rng(seed)
        self.prev_obs = None

    def discretize(self, observation, action=None, result=None):
        """将食堂观测离散化为16桶模式(b00-b15)"""
        if self.observer_type == 'structural':
            return self._discretize_structural(observation)
        elif self.observer_type == 'temporal':
            return self._discretize_temporal(observation)
        elif self.observer_type == 'functional':
            return self._discretize_functional(observation, action, result)
        return 'b07'

    def _discretize_structural(self, obs):
        """结构视角: 当前状态的离散化快照"""
        score = 0
        # 库存健康度 (0-5分)
        low_stock = obs.get('low_stock_count', 0)
        near_expiry = obs.get('near_expiry_count', 0)
        cert_expired = obs.get('cert_expired_count', 0)
        stock_health = max(0, 5 - low_stock - near_expiry - cert_expired)

        # 人员合规度 (0-5分)
        avg_skill = obs.get('avg_skill_level', 0.5)
        personnel_score = int(avg_skill * 5)

        # 设备状态 (0-3分)
        equip_status = obs.get('equipment_ok_ratio', 0.5)
        equip_score = int(equip_status * 3)

        # 预算充裕度 (0-3分)
        budget = obs.get('budget_ratio', 0.5)
        budget_score = int(budget * 3)

        combined = min(15, stock_health + personnel_score + equip_score + budget_score)
        return f"b{combined:02d}"

    def _discretize_temporal(self, obs):
        """时序视角: 与上次观测的变化模式"""
        if self.prev_obs is None:
            self.prev_obs = dict(obs)
            return 'b07'

        changes = 0
        # 检查各维度变化方向和幅度
        for key in ['low_stock_count', 'near_expiry_count', 'hidden_risk_count']:
            curr = obs.get(key, 0)
            prev = self.prev_obs.get(key, 0)
            if curr > prev:
                changes += 2  # 恶化
            elif curr < prev:
                changes -= 1  # 改善

        safety_curr = obs.get('safety_proxy', 0.5)
        safety_prev = self.prev_obs.get('safety_proxy', 0.5)
        if safety_curr < safety_prev - 0.05:
            changes += 3
        elif safety_curr > safety_prev + 0.05:
            changes -= 2

        self.prev_obs = dict(obs)

        # 映射到0-15: changes范围大约[-6, +12]
        bucket = max(0, min(15, changes + 6))
        return f"b{bucket:02d}"

    def _discretize_functional(self, obs, action=None, result=None):
        """功能视角: 行动-结果的因果模式"""
        if action is None or result is None:
            return 'b07'

        action_idx = {'inspect': 0, 'order': 1, 'menu_plan': 2,
                      'intervene': 3, 'staff_assign': 4, 'record': 5}.get(action, 0)

        success = 1 if result.get('success', False) else 0
        has_findings = 1 if result.get('findings') and len(result['findings']) > 0 else 0

        # 组合: action(6) × success(2) × findings(2) = 24 → 压缩到16
        raw = action_idx * 4 + success * 2 + has_findings
        # 加一点噪声模拟感知不确定性
        noise = self.rng.choice([0, 0, 0, 1])
        bucket = min(15, (raw + noise) % 16)
        return f"b{bucket:02d}"


# ============================================================
# C3共识知识库 (从v1.2移植, 适配食堂域)
# ============================================================
class ConsensusKnowledgeStore:
    SHARED_BUCKETS = [f'b{i:02d}' for i in range(16)]

    def __init__(self, min_observers=2, min_strength=0.15):
        self.candidates = {}
        self.confirmed = {}
        self.min_observers = min_observers
        self.min_strength = min_strength

    @staticmethod
    def to_shared_key(pivot_key, observer_type):
        if not isinstance(pivot_key, tuple) or len(pivot_key) < 3:
            return None
        in_pat, action, out_pat = pivot_key[0], pivot_key[1], pivot_key[2]
        in_bucket = str(in_pat) if str(in_pat).startswith('b') else 'b07'
        out_bucket = str(out_pat) if str(out_pat).startswith('b') else 'b07'
        return (in_bucket, action, out_bucket)

    def propose(self, pivot_key, observer_id, strength, action, observer_type='structural'):
        shared_key = self.to_shared_key(pivot_key, observer_type)
        if shared_key is None:
            return False

        if shared_key not in self.candidates:
            self.candidates[shared_key] = {
                'proposals': {},
                'action': action,
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
                    'action': action,
                    'confirmers': list(c['proposals'].keys()),
                    'cycle_completions': 1
                }
                return True
        return False

    def reinforce(self, pivot_key, success):
        if pivot_key in self.confirmed:
            c = self.confirmed[pivot_key]
            if success:
                c['strength'] = min(1.0, c['strength'] + 0.1)
                c['cycle_completions'] = c.get('cycle_completions', 0) + 1
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
                  'n_confirmers': v['n_confirmers'], 'action': v['action']}
                 for k, v in self.confirmed.items()],
                key=lambda x: x['strength'], reverse=True
            )[:8]
        }


# ============================================================
# 相变检测器 (从v1.2移植)
# ============================================================
class PhaseTransitionDetector:
    def __init__(self):
        self.betti_history = []
        self.last_pivot_keys = set()
        self.criticality_history = []
        self.transition_log = []

    def build_knowledge_graph(self, confirmed_pivots):
        G = nx.DiGraph()
        for key, val in confirmed_pivots.items():
            if isinstance(key, tuple) and len(key) >= 3:
                in_node = f"in_{key[0]}"
                out_node = f"out_{key[2]}"
                action = key[1]
                G.add_edge(in_node, out_node, action=action,
                           strength=val.get('strength', 0))
        return G

    def compute_betti(self, G):
        if len(G.nodes()) == 0:
            return {'b0': 0, 'b1': 0}
        b0 = nx.number_connected_components(G.to_undirected())
        try:
            cycles = nx.cycle_basis(G.to_undirected())
            b1 = len(cycles)
        except Exception:
            b1 = 0
        return {'b0': b0, 'b1': b1}

    def _compute_criticality(self, candidates):
        if not candidates:
            return 0
        min_obs = 2
        critical_count = 0
        for ck, cv in candidates.items():
            if cv.get('n_proposers', 0) == min_obs - 1:
                critical_count += 1
        return critical_count

    def check_raf_closure(self, confirmed_pivots):
        if not confirmed_pivots:
            return {'is_raf': False, 'coverage_ratio': 0.0}
        input_patterns = set()
        output_patterns = set()
        for key in confirmed_pivots:
            if isinstance(key, tuple) and len(key) >= 3:
                input_patterns.add(key[0])
                output_patterns.add(key[2])
        food_set = {'b07'}
        covered = 0
        total = len(input_patterns)
        for inp in input_patterns:
            if inp in output_patterns or inp in food_set:
                covered += 1
        ratio = covered / max(total, 1)
        return {'is_raf': ratio >= 0.8, 'coverage_ratio': ratio}

    def check_phase_transition(self, episode, confirmed_pivots, candidates=None):
        G = self.build_knowledge_graph(confirmed_pivots)
        betti = self.compute_betti(G)
        self.betti_history.append({'episode': episode, **betti})

        current_keys = set(confirmed_pivots.keys())
        added = current_keys - self.last_pivot_keys
        removed = self.last_pivot_keys - current_keys
        self.last_pivot_keys = current_keys

        criticality = self._compute_criticality(candidates) if candidates else 0
        self.criticality_history.append({'episode': episode, 'criticality': criticality})

        prev = self.betti_history[-2] if len(self.betti_history) >= 2 else None
        is_transition = False
        if prev and (betti['b0'] != prev['b0'] or betti['b1'] != prev['b1']):
            is_transition = True

        if is_transition:
            raf = self.check_raf_closure(confirmed_pivots)
            transition_info = {
                'episode': episode,
                'betti_before': prev,
                'betti_after': betti,
                'pivots_added': len(added),
                'pivots_removed': len(removed),
                'criticality': criticality,
                'raf_closure': raf
            }
            self.transition_log.append(transition_info)
            return transition_info
        return None


# ============================================================
# TUF-E 食堂智能体 (替换GroundedAgentV05)
# ============================================================
class TUFECafeteriaAgent:
    """
    用v1.2完整枢机机制驱动的食堂管理智能体。
    三个异构观察者各自独立感知 → C3共识确认枢机 → 枢机引导行为选择。
    """

    ACTIONS = ['inspect', 'order', 'menu_plan', 'intervene', 'staff_assign', 'record']

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.observers = {
            'structural': CafeteriaObserver('structural', seed=seed),
            'temporal': CafeteriaObserver('temporal', seed=seed + 1),
            'functional': CafeteriaObserver('functional', seed=seed + 2),
        }
        self.consensus_store = ConsensusKnowledgeStore(min_observers=2)
        self.phase_detector = PhaseTransitionDetector()

        self.action_preferences = {a: 1.0 for a in self.ACTIONS}
        self.private_pivots = {}
        self.self_model = {
            'known_issues': [],
            'confidence': 0.1,
            'issues_detected_count': 0,
            'successful_interventions': 0,
        }
        self.total_actions = 0
        self.successful_actions = 0
        self.recent_outcomes = []
        self.day_counter = 0

    def perceive_and_propose(self, observation, action, result):
        """三个观察者各自离散化并提议枢机"""
        situation_pat = self.observers['structural'].discretize(observation)
        trend_pat = self.observers['temporal'].discretize(observation)
        outcome_pat = self.observers['functional'].discretize(observation, action, result)

        success = result.get('success', False)
        strength = 0.3 if success else 0.1

        pivot_s = (situation_pat, action, outcome_pat)
        pivot_t = (trend_pat, action, outcome_pat)

        self.consensus_store.propose(pivot_s, 'structural', strength, action, 'structural')
        self.consensus_store.propose(pivot_t, 'temporal', strength, action, 'temporal')
        self.consensus_store.propose((situation_pat, action, trend_pat), 'functional', strength, action, 'functional')

        # 私有枢机记录
        priv_key = (situation_pat, action, outcome_pat)
        if priv_key in self.private_pivots:
            self.private_pivots[priv_key]['count'] += 1
            if success:
                self.private_pivots[priv_key]['cycle_completions'] = \
                    self.private_pivots[priv_key].get('cycle_completions', 0) + 1
        else:
            self.private_pivots[priv_key] = {'count': 1, 'cycle_completions': 1 if success else 0}

    def choose_action(self, observation, stage_info):
        available = stage_info['available_actions']
        confirmed = self.consensus_store.get_confirmed_pivots()

        prefs = np.array([max(0.05, self.action_preferences.get(a, 1.0)) for a in available])
        prefs /= prefs.sum()

        exploration = max(0.15, 1.0 - self.self_model['confidence'])

        # 枢机引导: 如果有匹配的已确认枢机，提高对应动作概率
        if confirmed and self.rng.random() > exploration:
            situation = self.observers['structural'].discretize(observation)
            for key, val in confirmed.items():
                if isinstance(key, tuple) and len(key) >= 3:
                    if key[0] == situation and key[1] in available:
                        idx = available.index(key[1])
                        prefs[idx] *= (1.0 + val['strength'])

        if self.rng.random() < exploration:
            action = self.rng.choice(available)
        else:
            prefs /= prefs.sum()
            action = self.rng.choice(available, p=prefs)

        params = self._generate_params(action, observation)
        return action, params

    def _generate_params(self, action, obs):
        if action == 'inspect':
            areas = [None, 'inventory', 'personnel', 'equipment', 'production']
            return {'area': self.rng.choice(areas)}
        elif action == 'order':
            return {
                'ingredient_id': int(self.rng.integers(0, 12)),
                'quantity': int(self.rng.integers(5, 30))
            }
        elif action == 'menu_plan':
            return {
                'budget_per_meal': float(self.rng.uniform(8, 15)),
                'taste_weight': float(self.rng.uniform(0.3, 0.8))
            }
        elif action == 'intervene':
            return {'finding_idx': int(self.rng.integers(0, max(1, len(self.self_model['known_issues']))))}
        return {}

    def learn(self, action, result, day_scores):
        self.total_actions += 1

        viability_signal = 0
        if result['success']:
            viability_signal += 0.2
            self.successful_actions += 1
        if 'findings' in result and len(result['findings']) > 0:
            self.self_model['known_issues'] = result['findings'][:3]
            self.self_model['issues_detected_count'] += len(result['findings'])
            diminishing = max(0.02, 0.15 / (1 + self.self_model['issues_detected_count'] / 50))
            viability_signal += diminishing

        if action == 'intervene' and result['success']:
            self.self_model['successful_interventions'] += 1
            viability_signal += 0.15

        if day_scores:
            viability_signal += day_scores.get('safety_score', 0.5) * 0.2
            if day_scores.get('compliance_score') is not None:
                viability_signal += day_scores['compliance_score'] * 0.2

        # 更新偏好(用viability而非reward)
        self.action_preferences[action] = max(0.05,
            self.action_preferences.get(action, 1.0) * (1.0 + 0.08 * viability_signal))
        total = sum(self.action_preferences.values())
        for k in self.action_preferences:
            self.action_preferences[k] /= total

        # 提议枢机 + 强化
        self.perceive_and_propose(
            self._last_observation if hasattr(self, '_last_observation') else {},
            action, result
        )

        confirmed = self.consensus_store.get_confirmed_pivots()
        for key in list(confirmed.keys()):
            if isinstance(key, tuple) and len(key) >= 3 and key[1] == action:
                self.consensus_store.reinforce(key, result['success'])

        self.recent_outcomes.append(viability_signal)
        if len(self.recent_outcomes) >= 10:
            recent_avg = np.mean(self.recent_outcomes[-20:])
            self.self_model['confidence'] = min(0.9, max(0.1, 0.3 + recent_avg))

    def end_of_day(self, day, day_scores):
        """每日结束时的维护操作"""
        self.day_counter = day
        self.consensus_store.decay(rate=0.003)

        confirmed = self.consensus_store.get_confirmed_pivots()
        candidates = self.consensus_store.candidates
        transition = self.phase_detector.check_phase_transition(day, confirmed, candidates)

        return transition

    def ready_for_stage(self, target_stage):
        if target_stage == 2:
            return self.self_model['issues_detected_count'] >= 30 and self.total_actions >= 40
        elif target_stage == 3:
            return (self.self_model['successful_interventions'] >= 15 and
                    self.self_model['issues_detected_count'] >= 60 and
                    self.total_actions >= 120)
        elif target_stage == 4:
            return self.total_actions >= 400 and self.self_model['issues_detected_count'] >= 200
        return False

    def get_stats(self):
        confirmed = self.consensus_store.get_confirmed_pivots()
        private_with_cycles = sum(1 for v in self.private_pivots.values()
                                   if v.get('cycle_completions', 0) >= 1)
        return {
            'total_actions': self.total_actions,
            'success_rate': round(self.successful_actions / max(self.total_actions, 1), 3),
            'n_confirmed_pivots': len(confirmed),
            'n_private_pivots': len(self.private_pivots),
            'n_private_with_cycles': private_with_cycles,
            'n_phase_transitions': len(self.phase_detector.transition_log),
            'confidence': round(self.self_model['confidence'], 3),
            'issues_detected': self.self_model['issues_detected_count'],
            'successful_interventions': self.self_model['successful_interventions'],
            'top_preferences': sorted(
                [(k, round(v, 3)) for k, v in self.action_preferences.items()],
                key=lambda x: x[1], reverse=True
            ),
            'consensus_summary': self.consensus_store.get_summary(),
        }


# ============================================================
# 实验主循环
# ============================================================
def run_experiment(n_days=120, n_seeds=5, report_every=10):
    print("=" * 70)
    print("TUF-E v1.3: 食堂管理域 + 完整枢机机制")
    print(f"规模: {n_seeds} seeds × {n_days} days")
    print("=" * 70)

    all_results = []

    for seed in range(n_seeds):
        env = CafeteriaEnvironment(seed=seed, n_days=n_days)
        agent = TUFECafeteriaAgent(seed=seed)

        transitions = []
        daily_data = []

        for day in range(n_days):
            obs = env.observe()
            agent._last_observation = obs
            stage_info = env.get_stage_info()

            if agent.ready_for_stage(env.stage + 1) and env.stage < 4:
                transition = env.try_advance_stage(True)
                if transition:
                    old_s, new_s = transition
                    transitions.append({
                        'day': day + 1,
                        'from': old_s, 'to': new_s,
                        'stats': agent.get_stats()
                    })

            actions_per_day = 4 if env.stage == 1 else 6
            for _ in range(actions_per_day):
                action, params = agent.choose_action(obs, stage_info)
                result = env.act(action, params)
                agent.learn(action, result, None)

            day_scores = env.advance_day()
            pt = agent.end_of_day(day, day_scores)

            if day_scores:
                daily_data.append({
                    'day': day + 1,
                    'safety': day_scores['safety_score'],
                    'compliance': day_scores.get('compliance_score'),
                    'satisfaction': day_scores['satisfaction'],
                    'cost_efficiency': max(0, day_scores['cost_efficiency']),
                    'hidden_risks': day_scores['hidden_risks'],
                    'phase_transition': pt is not None,
                })

            if (day + 1) % report_every == 0 and day_scores:
                stats = agent.get_stats()
                comply_str = f"{day_scores['compliance_score']:.2f}" if day_scores.get('compliance_score') is not None else "N/A"
                pt_mark = " ★" if pt else ""
                print(f"  S{seed} Day {day+1:>3} [{stage_info['name'][:4]}]: "
                      f"safety={day_scores['safety_score']:.2f} | "
                      f"comply={comply_str:>5} | "
                      f"pivots={stats['n_confirmed_pivots']:2d} | "
                      f"PT={stats['n_phase_transitions']:2d}{pt_mark}")

        final_stats = agent.get_stats()
        raf = agent.phase_detector.check_raf_closure(
            agent.consensus_store.get_confirmed_pivots())

        seed_result = {
            'seed': seed,
            'final_stats': final_stats,
            'transitions': transitions,
            'n_phase_transitions': final_stats['n_phase_transitions'],
            'raf_closure': raf,
            'daily_data': daily_data,
        }
        all_results.append(seed_result)

        # 计算阶段平均安全分
        stage_boundaries = [t['day'] for t in transitions]
        s1_end = stage_boundaries[0] if len(stage_boundaries) > 0 else 30
        s2_end = stage_boundaries[1] if len(stage_boundaries) > 1 else 60
        s3_end = stage_boundaries[2] if len(stage_boundaries) > 2 else 90

        safeties = [d['safety'] for d in daily_data]
        s1_safe = np.mean(safeties[:s1_end]) if s1_end <= len(safeties) else np.mean(safeties)
        s2_safe = np.mean(safeties[s1_end:s2_end]) if s2_end <= len(safeties) else 0
        s3_safe = np.mean(safeties[s2_end:s3_end]) if s3_end <= len(safeties) else 0
        s4_safe = np.mean(safeties[s3_end:]) if s3_end < len(safeties) else 0

        print(f"\n  Seed {seed} 总结: SR={final_stats['success_rate']:.3f} | "
              f"确认枢机={final_stats['n_confirmed_pivots']} | "
              f"顿悟={final_stats['n_phase_transitions']} | "
              f"RAF={'✓' if raf['is_raf'] else '✗'}({raf['coverage_ratio']:.2f})")
        print(f"    安全趋势: S1={s1_safe:.3f} → S2={s2_safe:.3f} → S3={s3_safe:.3f} → S4={s4_safe:.3f}")

    # === 汇总分析 ===
    print("\n" + "=" * 70)
    print("跨Seed汇总")
    print("=" * 70)

    avg_sr = np.mean([r['final_stats']['success_rate'] for r in all_results])
    avg_pivots = np.mean([r['final_stats']['n_confirmed_pivots'] for r in all_results])
    avg_pt = np.mean([r['n_phase_transitions'] for r in all_results])
    raf_count = sum(1 for r in all_results if r['raf_closure']['is_raf'])

    print(f"  平均成功率: {avg_sr:.3f}")
    print(f"  平均确认枢机: {avg_pivots:.1f}")
    print(f"  平均顿悟次数: {avg_pt:.1f}")
    print(f"  RAF达成率: {raf_count}/{n_seeds} ({raf_count/n_seeds:.0%})")

    # 安全趋势汇总
    all_s1, all_s2, all_s3, all_s4 = [], [], [], []
    for r in all_results:
        safeties = [d['safety'] for d in r['daily_data']]
        boundaries = [t['day'] for t in r['transitions']]
        s1e = boundaries[0] if len(boundaries) > 0 else 30
        s2e = boundaries[1] if len(boundaries) > 1 else 60
        s3e = boundaries[2] if len(boundaries) > 2 else 90
        all_s1.append(np.mean(safeties[:s1e]) if s1e <= len(safeties) else np.mean(safeties))
        all_s2.append(np.mean(safeties[s1e:s2e]) if s2e <= len(safeties) else 0)
        all_s3.append(np.mean(safeties[s2e:s3e]) if s3e <= len(safeties) else 0)
        all_s4.append(np.mean(safeties[s3e:]) if s3e < len(safeties) else 0)

    print(f"\n  安全分趋势 (均值±标准差):")
    print(f"    S1(观察):     {np.mean(all_s1):.3f} ± {np.std(all_s1):.3f}")
    print(f"    S2(规则内化): {np.mean(all_s2):.3f} ± {np.std(all_s2):.3f}")
    print(f"    S3(重组):     {np.mean(all_s3):.3f} ± {np.std(all_s3):.3f}")
    print(f"    S4(习惯退化): {np.mean(all_s4):.3f} ± {np.std(all_s4):.3f}")

    # 与v0.5对比
    print(f"\n  与v0.5基线对比:")
    print(f"    v0.5: SR≈0.95, 枢机=简单计数, 无相变检测, 无RAF验证")
    print(f"    v1.3: SR={avg_sr:.3f}, 枢机={avg_pivots:.1f}(C3共识), "
          f"顿悟={avg_pt:.1f}, RAF={raf_count/n_seeds:.0%}")

    # 保存结果
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('v13_cafeteria_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存至 v13_cafeteria_results.json")

    print("\n" + "=" * 70)
    print("实验完成")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment(n_days=120, n_seeds=5)
