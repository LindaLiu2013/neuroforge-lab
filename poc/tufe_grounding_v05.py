"""
TUF-E Grounded Agent v0.5: 食堂管理领域接地实验

设计原则(来自Jayson验证):
1. 验收门控是事后的(检查/事故时才暴露)，不是实时的
2. 菜单由经济逻辑驱动(利润+口味)，合规是边界条件
3. 台账≠实情(系统性偏差)
4. 失败模式是多因素耦合的，非单向排序

环境双层结构:
- 规范层: 法规/SOP(智能体可读)
- 操作层: 含人为偏差、经济压力、信息不完整(智能体需观察发现)

三阶段学习:
- 阶段1(观察): 学会识别不规范信号
- 阶段2(规则内化): 学会约束边界内的调整
- 阶段3(重组): 处理新型多因素场景
"""

import numpy as np
import json
import sys
import io
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# CAFETERIA ENVIRONMENT
# ============================================================
class CafeteriaEnvironment:
    """
    食堂管理模拟环境。

    状态空间: 7个维度
    - inventory: 库存状态(每种食材的量、温度、证照、保质期)
    - personnel: 人员状态(健康证、晨检、培训、岗位)
    - equipment: 设备状态(运行、温度、维护)
    - production: 生产状态(菜品进度、中心温度、保温时间)
    - compliance: 合规状态(记录完整性、检查结果)
    - financial: 财务状态(预算消耗、成本/餐)
    - satisfaction: 满意度(近期评分、投诉)

    动作空间: 6类决策
    - inspect: 验收/巡检(消耗时间，发现隐患)
    - order: 采购下单(受预算和供应商约束)
    - menu_plan: 菜单规划(受营养/成本/偏好约束)
    - staff_assign: 人员调配(受资质和健康约束)
    - intervene: 现场干预(纠正不规范)
    - record: 补录/更新台账

    反馈: 多维+时滞
    - safety_score: 安全评分(滞后1-3天暴露)
    - compliance_score: 合规评分(检查时才知道)
    - cost_efficiency: 成本效率(当日可知)
    - satisfaction: 满意度(次日反馈)
    """

    # 食材类别
    INGREDIENT_CATS = ['vegetable', 'meat', 'seafood', 'grain', 'oil_condiment', 'frozen']
    N_INGREDIENTS = 12  # 简化为12种代表性食材

    # 岗位
    STATIONS = ['receiving', 'storage', 'prep', 'cooking', 'serving', 'cleaning']

    # 检查项(对应真实的不规范识别点)
    INSPECTION_ITEMS = [
        'temp_violation',      # 温度超标(冷藏>8°C或热持<60°C)
        'cross_contamination', # 生熟混用工具
        'cert_missing',        # 证照缺失/过期
        'record_gap',          # 台账记录不完整
        'hygiene_lapse',       # 个人卫生不达标
        'fifo_violation',      # 先进先出未执行
        'sample_missing',      # 留样缺失/不足
        'equipment_fault',     # 设备异常未报修
        'staff_unfit',         # 人员带病上岗
        'waste_mismanage',     # 废弃物处置不当
    ]

    def __init__(self, seed=42, n_days=90):
        self.rng = np.random.default_rng(seed)
        self.n_days = n_days
        self.day = 0
        self.stage = 1  # 学习阶段

        # 隐藏真实状态(智能体不能直接读取)
        self._true_state = self._init_state()
        # 可观察状态(有噪声、有延迟、有不完整)
        self._observable_state = self._make_observable(self._true_state)

        # 事件队列(安全隐患→延迟暴露)
        self._pending_incidents = []
        # 检查计划
        self._next_inspection_day = self.rng.integers(5, 15)

        # 历史
        self.history = []
        self.daily_scores = {
            'safety': [], 'compliance': [], 'cost': [], 'satisfaction': []
        }

    def _init_state(self):
        state = {
            'inventory': {},
            'personnel': {},
            'equipment': {},
            'production': {'dishes_in_progress': [], 'samples_retained': []},
            'compliance': {'open_findings': [], 'last_inspection_day': -30},
            'financial': {'budget_remaining': 500000, 'daily_cost': 0, 'meals_served': 0},
            'satisfaction': {'recent_score': 0.75, 'complaints_today': 0},
            'hidden_risks': [],  # 已发生但未暴露的风险
        }

        for i in range(self.N_INGREDIENTS):
            cat = self.INGREDIENT_CATS[i % len(self.INGREDIENT_CATS)]
            state['inventory'][i] = {
                'category': cat,
                'quantity': float(self.rng.uniform(5, 50)),
                'temp': float(self.rng.uniform(0, 8)) if cat in ('meat', 'seafood', 'frozen') else 20.0,
                'has_cert': self.rng.random() > 0.1,
                'days_remaining': int(self.rng.integers(1, 14)),
                'received_verified': True,
                'actual_quality': float(self.rng.uniform(0.7, 1.0)),
            }

        for s in self.STATIONS:
            n_staff = int(self.rng.integers(2, 5))
            for j in range(n_staff):
                pid = f"{s}_{j}"
                state['personnel'][pid] = {
                    'station': s,
                    'health_cert_valid': self.rng.random() > 0.05,
                    'morning_check_passed': True,
                    'training_hours': float(self.rng.uniform(10, 40)),
                    'skill_level': float(self.rng.uniform(0.5, 1.0)),
                    'compliance_baseline': float(self.rng.uniform(0.4, 0.95)),
                    'fatigue': 0.0,
                    'supervision_sensitivity': float(self.rng.uniform(0.2, 0.8)),
                }

        for eq_type in ['fridge', 'freezer', 'stove', 'disinfection_cabinet']:
            state['equipment'][eq_type] = {
                'functional': self.rng.random() > 0.05,
                'temp_reading': float(self.rng.uniform(0, 8)) if 'fridge' in eq_type else None,
                'last_maintenance_days_ago': int(self.rng.integers(0, 60)),
            }

        return state

    def _make_observable(self, true_state):
        obs = {}
        obs['inventory_summary'] = {
            'total_items': len(true_state['inventory']),
            'low_stock_count': sum(1 for v in true_state['inventory'].values() if v['quantity'] < 10),
            'near_expiry_count': sum(1 for v in true_state['inventory'].values() if v['days_remaining'] <= 2),
        }
        obs['personnel_summary'] = {
            'total_staff': len(true_state['personnel']),
            'cert_expired_count': sum(1 for v in true_state['personnel'].values() if not v['health_cert_valid']),
            'avg_skill': round(np.mean([v['skill_level'] for v in true_state['personnel'].values()]), 2),
        }
        obs['equipment_summary'] = {
            'all_functional': all(v['functional'] for v in true_state['equipment'].values()),
            'maintenance_overdue': sum(1 for v in true_state['equipment'].values() if v['last_maintenance_days_ago'] > 30),
        }
        obs['financial'] = {
            'budget_remaining': true_state['financial']['budget_remaining'],
            'daily_cost': true_state['financial']['daily_cost'],
        }
        obs['satisfaction'] = {
            'recent_score': true_state['satisfaction']['recent_score'],
        }
        obs['day'] = self.day
        obs['stage'] = self.stage
        return obs

    def observe(self):
        return dict(self._observable_state)

    def deep_inspect(self, area=None):
        findings = []
        if area is None or area == 'inventory':
            for iid, inv in self._true_state['inventory'].items():
                if not inv['has_cert']:
                    findings.append(('cert_missing', f'ingredient_{iid}', 0.9))
                if inv['temp'] > 8 and inv['category'] in ('meat', 'seafood', 'frozen'):
                    findings.append(('temp_violation', f'ingredient_{iid}', 0.95))
                if inv['days_remaining'] <= 0:
                    findings.append(('fifo_violation', f'ingredient_{iid}', 0.85))
                if not inv['received_verified']:
                    findings.append(('record_gap', f'ingredient_{iid}', 0.8))

        if area is None or area == 'personnel':
            for pid, p in self._true_state['personnel'].items():
                if not p['health_cert_valid']:
                    findings.append(('cert_missing', pid, 0.9))
                eff_comp = p.get('_effective_compliance_today', p['compliance_baseline'])
                if self.rng.random() > eff_comp:
                    findings.append(('hygiene_lapse', pid, round(1.0 - eff_comp, 2)))

        if area is None or area == 'equipment':
            for eq, e in self._true_state['equipment'].items():
                if not e['functional']:
                    findings.append(('equipment_fault', eq, 0.95))
                if e['temp_reading'] is not None and e['temp_reading'] > 8:
                    findings.append(('temp_violation', eq, 0.9))

        if area is None or area == 'production':
            if self.rng.random() < 0.3:
                findings.append(('sample_missing', 'kitchen', 0.75))
            if self.rng.random() < 0.2:
                findings.append(('cross_contamination', 'prep_area', 0.65))

        return findings

    def act(self, action_type, params=None):
        result = {'action': action_type, 'success': False, 'feedback': '', 'cost': 0}

        if action_type == 'inspect':
            area = params.get('area') if params else None
            findings = self.deep_inspect(area)
            result['findings'] = findings
            result['success'] = True
            result['cost'] = 50
            result['feedback'] = f"Found {len(findings)} issues"
            for item_type, location, severity in findings:
                self._true_state['compliance']['open_findings'].append({
                    'type': item_type, 'location': location,
                    'severity': severity, 'found_day': self.day
                })

        elif action_type == 'order':
            ingredient_id = params.get('ingredient_id', 0) if params else 0
            quantity = params.get('quantity', 20) if params else 20
            cost = quantity * float(self.rng.uniform(3, 15))
            if cost <= self._true_state['financial']['budget_remaining']:
                self._true_state['financial']['budget_remaining'] -= cost
                if ingredient_id in self._true_state['inventory']:
                    self._true_state['inventory'][ingredient_id]['quantity'] += quantity
                    self._true_state['inventory'][ingredient_id]['days_remaining'] = int(self.rng.integers(3, 10))
                    verify_rate = 0.7 if self.stage >= 2 else 0.4
                    self._true_state['inventory'][ingredient_id]['received_verified'] = self.rng.random() < verify_rate
                    self._true_state['inventory'][ingredient_id]['has_cert'] = self.rng.random() < 0.85
                result['success'] = True
                result['cost'] = cost
                result['feedback'] = f"Ordered {quantity} units, cost={cost:.0f}"
            else:
                result['feedback'] = "Budget exceeded"

        elif action_type == 'intervene':
            finding_idx = params.get('finding_idx', 0) if params else 0
            findings = self._true_state['compliance']['open_findings']
            if finding_idx < len(findings):
                finding = findings[finding_idx]
                fix_success = self.rng.random() < (0.8 if self.stage >= 2 else 0.5)
                if fix_success:
                    findings.pop(finding_idx)
                    result['success'] = True
                    result['feedback'] = f"Fixed {finding['type']} at {finding['location']}"
                else:
                    result['feedback'] = f"Intervention failed for {finding['type']}"
            else:
                result['feedback'] = "No finding at that index"

        elif action_type == 'menu_plan':
            budget_per_meal = params.get('budget_per_meal', 12) if params else 12
            taste_weight = params.get('taste_weight', 0.6) if params else 0.6
            nutrition_weight = 1.0 - taste_weight
            sat_impact = taste_weight * 0.1 + nutrition_weight * 0.05
            cost_impact = budget_per_meal * 300
            if cost_impact <= self._true_state['financial']['budget_remaining']:
                self._true_state['satisfaction']['recent_score'] = min(1.0,
                    self._true_state['satisfaction']['recent_score'] + sat_impact * float(self.rng.uniform(0.5, 1.5)))
                self._true_state['financial']['daily_cost'] = cost_impact
                result['success'] = True
                result['cost'] = cost_impact
                result['feedback'] = f"Menu planned: taste_w={taste_weight:.1f}, nutr_w={nutrition_weight:.1f}"
            else:
                result['feedback'] = "Menu too expensive for remaining budget"

        elif action_type == 'staff_assign':
            result['success'] = True
            result['feedback'] = "Staff reassigned"

        elif action_type == 'record':
            result['success'] = True
            result['feedback'] = "Records updated"
            if self.rng.random() < 0.3:
                self._true_state['hidden_risks'].append({
                    'type': 'record_gap',
                    'day_created': self.day,
                    'exposure_delay': int(self.rng.integers(3, 15))
                })

        return result

    def advance_day(self):
        self.day += 1

        for inv in self._true_state['inventory'].values():
            inv['days_remaining'] -= 1
            if inv['category'] in ('meat', 'seafood', 'frozen'):
                if self.rng.random() < 0.05:
                    inv['temp'] += float(self.rng.uniform(1, 4))
            if inv['days_remaining'] <= 0 and self.rng.random() < 0.3:
                inv['actual_quality'] *= 0.7

        for p in self._true_state['personnel'].values():
            p['morning_check_passed'] = self.rng.random() > 0.03
            p['fatigue'] = min(1.0, p['fatigue'] + float(self.rng.uniform(0.02, 0.08)))
            if self.rng.random() < 0.1:
                p['fatigue'] *= 0.5
            inspection_recent = (self.day - self._true_state['compliance']['last_inspection_day']) <= 3
            supervision_boost = 0.2 * p['supervision_sensitivity'] if inspection_recent else 0
            effective_compliance = max(0.1, min(1.0,
                p['compliance_baseline'] - p['fatigue'] * 0.3 + supervision_boost))
            p['_effective_compliance_today'] = effective_compliance

        for e in self._true_state['equipment'].values():
            if e['functional'] and self.rng.random() < 0.02:
                e['functional'] = False
            e['last_maintenance_days_ago'] += 1

        daily_meals = int(self.rng.integers(280, 350))
        self._true_state['financial']['meals_served'] += daily_meals
        base_cost = daily_meals * float(self.rng.uniform(10, 14))
        self._true_state['financial']['daily_cost'] = base_cost
        self._true_state['financial']['budget_remaining'] -= base_cost

        risk_exposure_chance = 0.15 if self.stage == 1 else (0.25 if self.stage == 2 else 0.35)
        newly_exposed = []
        remaining_risks = []
        for risk in self._true_state['hidden_risks']:
            if self.day - risk['day_created'] >= risk['exposure_delay']:
                if self.rng.random() < risk_exposure_chance:
                    newly_exposed.append(risk)
                else:
                    remaining_risks.append(risk)
            else:
                remaining_risks.append(risk)
        self._true_state['hidden_risks'] = remaining_risks

        if self.day >= self._next_inspection_day:
            inspection_findings = self.deep_inspect()
            n_findings = len(inspection_findings)
            compliance_score = max(0, 1.0 - n_findings * 0.1)
            self._true_state['compliance']['last_inspection_day'] = self.day
            self._next_inspection_day = self.day + int(self.rng.integers(7, 21))
            newly_exposed.extend([{
                'type': f'inspection_{f[0]}', 'day_created': self.day, 'exposure_delay': 0
            } for f in inspection_findings])
        else:
            compliance_score = None

        n_active_risks = len(self._true_state['hidden_risks']) + len(newly_exposed)
        safety_score = max(0, 1.0 - n_active_risks * 0.08)

        sat_noise = float(self.rng.normal(0, 0.05))
        self._true_state['satisfaction']['recent_score'] = max(0.1, min(1.0,
            self._true_state['satisfaction']['recent_score'] + sat_noise))
        satisfaction = self._true_state['satisfaction']['recent_score']

        cost_efficiency = min(1.0, self._true_state['financial']['budget_remaining'] / 100000)

        day_result = {
            'day': self.day,
            'safety_score': round(safety_score, 3),
            'compliance_score': round(compliance_score, 3) if compliance_score is not None else None,
            'cost_efficiency': round(cost_efficiency, 3),
            'satisfaction': round(satisfaction, 3),
            'risks_exposed': len(newly_exposed),
            'hidden_risks': len(self._true_state['hidden_risks']),
            'budget_remaining': round(self._true_state['financial']['budget_remaining'], 0),
        }

        self.daily_scores['safety'].append(safety_score)
        if compliance_score is not None:
            self.daily_scores['compliance'].append(compliance_score)
        self.daily_scores['cost'].append(cost_efficiency)
        self.daily_scores['satisfaction'].append(satisfaction)

        self._observable_state = self._make_observable(self._true_state)

        stage_thresholds = {1: 30, 2: 60}
        if self.day >= stage_thresholds.get(self.stage, 999):
            self.stage += 1

        self.apply_habituaton_effect()

        self.history.append(day_result)
        return day_result

    def try_advance_stage(self, agent_ready):
        if agent_ready and self.stage < 4:
            old = self.stage
            self.stage += 1
            return old, self.stage
        return None

    def apply_habituaton_effect(self):
        if self.stage >= 4:
            for p in self._true_state['personnel'].values():
                p['compliance_baseline'] = max(0.3, p['compliance_baseline'] - 0.003)
            inspection_age = self.day - self._true_state['compliance']['last_inspection_day']
            if inspection_age > 14:
                vigilance_decay = min(0.3, inspection_age * 0.01)
                for inv in self._true_state['inventory'].values():
                    if self.rng.random() < vigilance_decay * 0.1:
                        inv['received_verified'] = False
                if self.rng.random() < vigilance_decay * 0.15:
                    self._true_state['hidden_risks'].append({
                        'type': 'habituation_blind_spot',
                        'day_created': self.day,
                        'exposure_delay': int(self.rng.integers(5, 20))
                    })

    def get_stage_info(self):
        if self.stage == 1:
            return {
                'name': 'OBSERVATION',
                'description': 'Learn to detect non-compliance signals',
                'available_actions': ['inspect', 'observe'],
                'focus': 'Identify what is wrong before trying to fix it'
            }
        elif self.stage == 2:
            return {
                'name': 'RULE_INTERNALIZATION',
                'description': 'Learn constraint-boundary adjustment',
                'available_actions': ['inspect', 'order', 'menu_plan', 'intervene', 'staff_assign', 'record'],
                'focus': 'Balance compliance, cost, and satisfaction within constraints'
            }
        elif self.stage == 3:
            return {
                'name': 'RECOMPOSITION',
                'description': 'Handle novel multi-factor scenarios',
                'available_actions': ['inspect', 'order', 'menu_plan', 'intervene', 'staff_assign', 'record'],
                'focus': 'Deal with coupled failures and unexpected events'
            }
        else:
            return {
                'name': 'HABITUATION_DECLINE',
                'description': 'Competence erodes through routine; external triggers needed to re-activate vigilance',
                'available_actions': ['inspect', 'order', 'menu_plan', 'intervene', 'staff_assign', 'record'],
                'focus': 'Detect and counteract own complacency'
            }


# ============================================================
# TUF-E GROUNDED AGENT v0.5
# ============================================================
class GroundedAgentV05:
    OP_NAMES = {
        'inspect': '巡检/验收',
        'order': '采购下单',
        'menu_plan': '菜单规划',
        'intervene': '现场干预',
        'staff_assign': '人员调配',
        'record': '台账记录',
    }

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.action_preferences = {
            'inspect': 1.0, 'order': 1.0, 'menu_plan': 1.0,
            'intervene': 1.0, 'staff_assign': 1.0, 'record': 1.0
        }
        self.pivot_library = {}
        self.self_model = {
            'known_issues': [],
            'confidence': 0.1,
            'stage_understanding': 'unknown',
            'issues_detected_count': 0,
            'successful_interventions': 0,
        }
        self.total_actions = 0
        self.successful_actions = 0
        self.energy_bits = 0
        self.recent_outcomes = []

    def choose_action(self, observation, stage_info):
        available = stage_info['available_actions']
        prefs = np.array([max(0.05, self.action_preferences.get(a, 1.0)) for a in available])
        prefs /= prefs.sum()

        exploration = max(0.15, 1.0 - self.self_model['confidence'])
        if self.rng.random() < exploration:
            action = self.rng.choice(available)
        else:
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
            taste_w = float(self.rng.uniform(0.3, 0.8))
            return {
                'budget_per_meal': float(self.rng.uniform(8, 15)),
                'taste_weight': taste_w
            }
        elif action == 'intervene':
            return {'finding_idx': int(self.rng.integers(0, max(1, len(self.self_model['known_issues']))))}
        return {}

    def learn(self, action, result, day_scores):
        self.total_actions += 1
        self.energy_bits += 256

        reward = 0
        if result['success']:
            reward += 0.2
            self.successful_actions += 1
        if 'findings' in result and len(result['findings']) > 0:
            new_findings = len(result['findings'])
            self.self_model['known_issues'] = result['findings'][:3]
            self.self_model['issues_detected_count'] += new_findings
            diminishing = max(0.02, 0.15 / (1 + self.self_model['issues_detected_count'] / 50))
            reward += diminishing

        if action == 'intervene' and result['success']:
            self.self_model['successful_interventions'] += 1
            n_open = len(self.self_model['known_issues'])
            urgency_bonus = 0.25 if n_open > 3 else (0.15 if n_open > 0 else 0.05)
            reward += urgency_bonus

        if action == 'menu_plan' and result['success']:
            reward += 0.15
        if action == 'order' and result['success']:
            reward += 0.12
        if action == 'staff_assign' and result['success']:
            reward += 0.12
        if action == 'record' and result['success']:
            reward += 0.08

        if action == 'intervene' and not result['success']:
            reward -= 0.05

        if day_scores:
            reward += day_scores.get('safety_score', 0.5) * 0.2
            if day_scores.get('compliance_score') is not None:
                reward += day_scores['compliance_score'] * 0.2
            reward += day_scores.get('satisfaction', 0.5) * 0.1

        self.action_preferences[action] = max(0.05, self.action_preferences.get(action, 1.0) * (1.0 + 0.08 * reward))
        total = sum(self.action_preferences.values())
        for k in self.action_preferences:
            self.action_preferences[k] /= total

        pivot_key = (action, result['success'], 'good' if reward > 0.3 else 'poor')
        if pivot_key in self.pivot_library:
            self.pivot_library[pivot_key]['count'] += 1
            self.pivot_library[pivot_key]['avg_reward'] = (
                self.pivot_library[pivot_key]['avg_reward'] * 0.9 + reward * 0.1
            )
        else:
            self.pivot_library[pivot_key] = {'count': 1, 'avg_reward': reward}

        self.recent_outcomes.append(reward)
        if len(self.recent_outcomes) >= 10:
            recent_avg = np.mean(self.recent_outcomes[-20:])
            self.self_model['confidence'] = min(0.9, max(0.1, 0.3 + recent_avg))

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
        return {
            'total_actions': self.total_actions,
            'success_rate': round(self.successful_actions / max(self.total_actions, 1), 3),
            'n_pivots': len(self.pivot_library),
            'confidence': round(self.self_model['confidence'], 3),
            'energy_bits': self.energy_bits,
            'top_preferences': sorted(
                [(k, round(v, 3)) for k, v in self.action_preferences.items()],
                key=lambda x: x[1], reverse=True
            ),
            'top_pivots': sorted(
                [{'action': k[0], 'success': k[1], 'quality': k[2],
                  'count': v['count'], 'avg_reward': round(v['avg_reward'], 3)}
                 for k, v in self.pivot_library.items()],
                key=lambda x: x['count'], reverse=True
            )[:8]
        }


# ============================================================
# EXPERIMENT
# ============================================================
def run_v05_experiment(n_days=120, report_every=10):
    print("=" * 70)
    print("TUF-E Grounded Agent v0.5: Cafeteria Management Domain")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    env = CafeteriaEnvironment(seed=42, n_days=n_days)
    agent = GroundedAgentV05(seed=42)

    results = {'milestones': [], 'final': {}, 'stage_transitions': []}

    print(f"\nDomain: Vocational school cafeteria management")
    print(f"Duration: {n_days} days, 4 learning stages")
    print(f"Stage 1 (Observation): Learn to see problems")
    print(f"Stage 2 (Rule Internalization): Balance constraints")
    print(f"Stage 3 (Recomposition): Handle coupled failures")
    print(f"Stage 4 (Habituaton Decline): Competence erodes through routine")

    current_stage = 1

    for day in range(n_days):
        obs = env.observe()
        stage_info = env.get_stage_info()

        if agent.ready_for_stage(env.stage + 1) and env.stage < 4:
            transition = env.try_advance_stage(True)
            if transition:
                old_s, new_s = transition
                print(f"\n  >>> STAGE TRANSITION at day {day+1}: Stage {old_s}→{new_s} ({env.get_stage_info()['name']})")
                print(f"      Issues detected: {agent.self_model['issues_detected_count']}, "
                      f"Interventions: {agent.self_model['successful_interventions']}")
                results['stage_transitions'].append({
                    'day': day + 1,
                    'from_stage': old_s,
                    'to_stage': new_s,
                    'agent_stats_at_transition': agent.get_stats()
                })
                current_stage = new_s
                stage_info = env.get_stage_info()

        actions_per_day = 4 if env.stage == 1 else 6
        day_scores = None

        for _ in range(actions_per_day):
            action, params = agent.choose_action(obs, stage_info)
            result = env.act(action, params)
            agent.learn(action, result, None)

        day_scores = env.advance_day()

        if day_scores:
            agent.recent_outcomes.append(
                day_scores['safety_score'] * 0.3 +
                (day_scores['compliance_score'] if day_scores.get('compliance_score') else 0.5) * 0.3 +
                day_scores['satisfaction'] * 0.2 +
                max(0, day_scores['cost_efficiency']) * 0.2
            )

        if (day + 1) % report_every == 0 and day_scores:
            stats = agent.get_stats()
            comply_str = f"{day_scores['compliance_score']:.2f}" if day_scores.get('compliance_score') is not None else "N/A"
            print(f"  Day {day+1:>3}/{n_days} [{stage_info['name'][:4]}]: "
                  f"safety={day_scores['safety_score']:.2f} | "
                  f"comply={comply_str:>5} | "
                  f"cost={max(0,day_scores['cost_efficiency']):.2f} | "
                  f"sat={day_scores['satisfaction']:.2f} | "
                  f"hidden={day_scores['hidden_risks']} | "
                  f"agent_sr={stats['success_rate']:.2f} | "
                  f"conf={stats['confidence']:.2f}")

            results['milestones'].append({
                'day': day + 1,
                'stage': stage_info['name'],
                'scores': day_scores,
                'agent_stats': stats
            })

    final_stats = agent.get_stats()
    s1_end = results['stage_transitions'][0]['day'] if len(results['stage_transitions']) > 0 else 30
    s2_end = results['stage_transitions'][1]['day'] if len(results['stage_transitions']) > 1 else 60
    s3_end = results['stage_transitions'][2]['day'] if len(results['stage_transitions']) > 2 else 90
    results['final'] = {
        'agent_stats': final_stats,
        'score_trends': {
            'avg_safety_stage1': round(float(np.mean(env.daily_scores['safety'][:s1_end])), 3),
            'avg_safety_stage2': round(float(np.mean(env.daily_scores['safety'][s1_end:s2_end])), 3),
            'avg_safety_stage3': round(float(np.mean(env.daily_scores['safety'][s2_end:s3_end])), 3),
            'avg_safety_stage4': round(float(np.mean(env.daily_scores['safety'][s3_end:])), 3) if s3_end < n_days else None,
            'avg_satisfaction_stage1': round(float(np.mean(env.daily_scores['satisfaction'][:s1_end])), 3),
            'avg_satisfaction_stage2': round(float(np.mean(env.daily_scores['satisfaction'][s1_end:s2_end])), 3),
            'avg_satisfaction_stage3': round(float(np.mean(env.daily_scores['satisfaction'][s2_end:s3_end])), 3),
            'avg_satisfaction_stage4': round(float(np.mean(env.daily_scores['satisfaction'][s3_end:])), 3) if s3_end < n_days else None,
        },
        'final_budget': round(env._true_state['financial']['budget_remaining'], 0),
    }

    print(f"\n{'=' * 70}")
    print("Final Results")
    print(f"{'=' * 70}")
    ts = results['final']['score_trends']
    s4_safety = f"{ts['avg_safety_stage4']:.3f}" if ts.get('avg_safety_stage4') is not None else "N/A"
    s4_sat = f"{ts['avg_satisfaction_stage4']:.3f}" if ts.get('avg_satisfaction_stage4') is not None else "N/A"
    print(f"  Safety trend:       S1={ts['avg_safety_stage1']:.3f} → S2={ts['avg_safety_stage2']:.3f} → S3={ts['avg_safety_stage3']:.3f} → S4={s4_safety}")
    print(f"  Satisfaction trend: S1={ts['avg_satisfaction_stage1']:.3f} → S2={ts['avg_satisfaction_stage2']:.3f} → S3={ts['avg_satisfaction_stage3']:.3f} → S4={s4_sat}")
    print(f"  Final budget: {results['final']['final_budget']:.0f}")
    print(f"  Agent success rate: {final_stats['success_rate']:.3f}")
    print(f"  Pivots learned: {final_stats['n_pivots']}")
    print(f"  Confidence: {final_stats['confidence']:.3f}")

    print(f"\n  Action preferences:")
    for action, pref in final_stats['top_preferences']:
        print(f"    {action}: {pref:.3f}")

    print(f"\n  Top pivots:")
    for p in final_stats['top_pivots'][:5]:
        print(f"    {p['action']}({'✓' if p['success'] else '✗'},{p['quality']}): count={p['count']}, reward={p['avg_reward']:.3f}")

    if results['stage_transitions']:
        print(f"\n  Stage transitions:")
        for st in results['stage_transitions']:
            print(f"    Day {st['day']}: Stage {st['from_stage']}→{st['to_stage']}, "
                  f"SR={st['agent_stats_at_transition']['success_rate']:.3f}, "
                  f"pivots={st['agent_stats_at_transition']['n_pivots']}")

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('grounding_v05_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_v05_results.json")

    return results


if __name__ == '__main__':
    run_v05_experiment(n_days=120, report_every=10)
