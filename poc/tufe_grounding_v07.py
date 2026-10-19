"""
TUF-E Grounded Agent v0.7: Triple-Observer Hybrid (LLM + Statistical + TUF-E)

v0.6发现: LLM对慢变量(合规侵蚀/记录缺口/检查间隔效应)检出率仅42%。
v0.7新增: 非语言统计感知通道，专门监测时间序列趋势和分布漂移。

三个异构观察者:
1. LLM感知层: 叙事性解读，擅长紧急事件，盲于缓慢退化
2. 统计感知层: 时序分析，擅长趋势/漂移/周期性，无语义理解
3. TUF-E认知层: 拓扑结构分析，擅长关系模式，依赖前两者提供原始信号

核心问题: 三通道共识是否比双通道更好地补偿各自的盲区？
"""

import numpy as np
import json
import sys
import io
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# CAFETERIA ENVIRONMENT (reused from v0.5)
# ============================================================
class CafeteriaEnvironment:
    INGREDIENT_CATS = ['meat', 'vegetable', 'seafood', 'grain', 'frozen', 'condiment']
    STATIONS = ['prep', 'cook', 'serve', 'wash']
    N_INGREDIENTS = 12

    def __init__(self, seed=42, n_days=120):
        self.rng = np.random.default_rng(seed)
        self.n_days = n_days
        self.day = 0
        self.stage = 1
        self._true_state = self._init_state()
        self._observable_state = self._make_observable(self._true_state)
        self._next_inspection_day = int(self.rng.integers(7, 14))
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
            'hidden_risks': [],
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

    def get_raw_data_for_llm(self):
        """Return richer raw data that an LLM would receive as text input."""
        raw = {
            'day': self.day,
            'stage': self.stage,
            'inventory_details': [],
            'personnel_details': [],
            'equipment_status': [],
            'recent_findings': self._true_state['compliance']['open_findings'][-5:],
            'financial': dict(self._true_state['financial']),
            'satisfaction': dict(self._true_state['satisfaction']),
        }
        for iid, inv in self._true_state['inventory'].items():
            raw['inventory_details'].append({
                'id': iid, 'category': inv['category'],
                'quantity': round(inv['quantity'], 1),
                'temp': round(inv['temp'], 1),
                'has_cert': inv['has_cert'],
                'days_remaining': inv['days_remaining'],
                'verified': inv['received_verified'],
            })
        for pid, p in self._true_state['personnel'].items():
            raw['personnel_details'].append({
                'id': pid, 'station': p['station'],
                'cert_valid': p['health_cert_valid'],
                'morning_check': p['morning_check_passed'],
                'skill': round(p['skill_level'], 2),
                'fatigue': round(p['fatigue'], 2),
            })
        for eq, e in self._true_state['equipment'].items():
            raw['equipment_status'].append({
                'type': eq, 'functional': e['functional'],
                'temp': round(e['temp_reading'], 1) if e['temp_reading'] else None,
                'maintenance_overdue': e['last_maintenance_days_ago'] > 30,
            })
        return raw

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
                'description': 'Competence erodes through routine',
                'available_actions': ['inspect', 'order', 'menu_plan', 'intervene', 'staff_assign', 'record'],
                'focus': 'Detect and counteract own complacency'
            }


# ============================================================
# SIMULATED LLM PERCEPTION FUNCTION
# ============================================================
class LLMPerceptionFunction:
    """
    Simulates how an LLM perceives structured cafeteria data.

    Key biases modeled (based on real LLM behavior):
    1. Salience bias: Over-reports dramatic/urgent issues, under-reports slow drift
    2. Narrative consistency: Tries to construct coherent story, may smooth contradictions
    3. Recency bias: Recent events weighted more than older ones
    4. Numerical imprecision: Converts exact numbers to qualitative ranges
    5. Confidence miscalibration: High confidence even when wrong
    6. Category framing: Interprets data through pre-existing conceptual categories
    """

    ISSUE_TEMPLATES = {
        'cert_missing': [
            "发现{loc}证照缺失，存在合规风险",
            "{loc}的资质文件不完整，需要立即补充",
            "注意到{loc}相关证件未更新",
        ],
        'temp_violation': [
            "⚠️ {loc}温度异常！当前读数偏高，食品安全受威胁",
            "{loc}冷链可能断裂，温度超标需紧急处理",
            "检测到{loc}温控异常，建议立即排查",
        ],
        'hygiene_lapse': [
            "{loc}人员操作不够规范，卫生习惯需加强",
            "观察到{loc}在操作流程中有疏忽",
            "{loc}的个人卫生执行不到位",
        ],
        'equipment_fault': [
            "🔧 {loc}设备故障！影响正常生产",
            "{loc}运行异常，可能需要维修",
            "{loc}出现故障信号",
        ],
        'fifo_violation': [
            "{loc}库存周转不规范，先进先出原则未执行",
            "发现{loc}有过期风险物料未清理",
        ],
        'record_gap': [
            "{loc}台账记录不完整",
            "{loc}的记录与实际不符",
        ],
        'sample_missing': [
            "留样制度执行不到位",
            "今日留样记录缺失",
        ],
        'cross_contamination': [
            "⚠️ 加工区域存在交叉污染风险！",
            "生熟分区执行不严格",
        ],
    }

    SALIENCE_WEIGHTS = {
        'temp_violation': 3.0,
        'equipment_fault': 2.5,
        'cross_contamination': 2.8,
        'cert_missing': 1.5,
        'hygiene_lapse': 1.2,
        'fifo_violation': 1.0,
        'record_gap': 0.6,
        'sample_missing': 0.8,
    }

    SLOW_DRIFT_DETECTION_RATE = 0.3

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.perception_history = []
        self.total_perceptions = 0
        self.accurate_perceptions = 0
        self.missed_issues = 0
        self.false_alarms = 0

    def perceive(self, raw_data, true_findings):
        """
        Transform raw data into LLM-style perception report.
        Returns: structured perception dict with biases applied.
        """
        self.total_perceptions += 1
        perceived_issues = []
        missed = []
        false_positives = []

        for finding_type, location, severity in true_findings:
            salience = self.SALIENCE_WEIGHTS.get(finding_type, 1.0)
            detection_prob = min(0.95, 0.3 + salience * 0.2)

            if finding_type in ('record_gap', 'fifo_violation'):
                detection_prob *= self.SLOW_DRIFT_DETECTION_RATE

            if self.rng.random() < detection_prob:
                templates = self.ISSUE_TEMPLATES.get(finding_type, ["发现{loc}存在问题"])
                narrative = self.rng.choice(templates).format(loc=location)
                perceived_severity = self._distort_severity(severity, finding_type)
                perceived_issues.append({
                    'type': finding_type,
                    'location': location,
                    'narrative': narrative,
                    'perceived_severity': perceived_severity,
                    'true_severity': severity,
                    'confidence': self._generate_confidence(perceived_severity),
                })
                self.accurate_perceptions += 1
            else:
                missed.append({'type': finding_type, 'location': location})
                self.missed_issues += 1

        if self.rng.random() < 0.08:
            false_type = self.rng.choice(['hygiene_lapse', 'record_gap', 'temp_violation'])
            false_loc = self.rng.choice(['prep_area', 'storage', 'cook_station'])
            templates = self.ISSUE_TEMPLATES.get(false_type, ["{loc}疑似异常"])
            false_positives.append({
                'type': false_type,
                'location': false_loc,
                'narrative': self.rng.choice(templates).format(loc=false_loc),
                'perceived_severity': float(self.rng.uniform(0.3, 0.7)),
                'true_severity': 0.0,
                'confidence': float(self.rng.uniform(0.5, 0.85)),
                'is_false_positive': True,
            })
            self.false_alarms += 1

        trend_signals = self._extract_trends(raw_data)
        overall_assessment = self._generate_assessment(perceived_issues, raw_data)

        perception = {
            'issues': perceived_issues + false_positives,
            'missed': missed,
            'trend_signals': trend_signals,
            'overall_assessment': overall_assessment,
            'n_detected': len(perceived_issues),
            'n_missed': len(missed),
            'n_false': len(false_positives),
            'accuracy': len(perceived_issues) / max(len(true_findings), 1),
        }

        self.perception_history.append(perception)
        return perception

    def _distort_severity(self, true_severity, issue_type):
        salience = self.SALIENCE_WEIGHTS.get(issue_type, 1.0)
        if salience > 2.0:
            distorted = min(1.0, true_severity * float(self.rng.uniform(1.1, 1.4)))
        elif salience < 0.8:
            distorted = true_severity * float(self.rng.uniform(0.5, 0.9))
        else:
            distorted = true_severity * float(self.rng.uniform(0.8, 1.2))
        return round(max(0.0, min(1.0, distorted)), 2)

    def _generate_confidence(self, severity):
        base = 0.6 + severity * 0.3
        noise = float(self.rng.normal(0, 0.1))
        return round(max(0.2, min(0.99, base + noise)), 2)

    def _extract_trends(self, raw_data):
        signals = []
        budget = raw_data['financial']['budget_remaining']
        if budget < 100000:
            signals.append({
                'dimension': 'budget',
                'signal': 'critical' if budget < 50000 else 'warning',
                'narrative': f"预算余额¥{budget:.0f}，{'严重不足' if budget < 50000 else '偏低'}",
            })
        sat = raw_data['satisfaction']['recent_score']
        if sat < 0.6:
            signals.append({
                'dimension': 'satisfaction',
                'signal': 'declining',
                'narrative': f"满意度降至{sat:.2f}，需要关注菜品质量",
            })
        elif sat > 0.9:
            signals.append({
                'dimension': 'satisfaction',
                'signal': 'high',
                'narrative': f"满意度良好({sat:.2f})",
            })
        return signals

    def _generate_assessment(self, issues, raw_data):
        n_issues = len(issues)
        if n_issues == 0:
            return {
                'risk_level': 'low',
                'summary': '当前运营状态正常，未发现明显问题',
                'recommended_focus': 'routine_maintenance',
            }
        high_sev = sum(1 for i in issues if i.get('perceived_severity', 0) > 0.7)
        if high_sev >= 2:
            return {
                'risk_level': 'high',
                'summary': f'发现{high_sev}个高优先级问题，需要立即处理',
                'recommended_focus': 'urgent_intervention',
            }
        elif n_issues >= 3:
            return {
                'risk_level': 'medium',
                'summary': f'累计{n_issues}个问题待处理，建议系统性排查',
                'recommended_focus': 'systematic_inspection',
            }
        else:
            return {
                'risk_level': 'low',
                'summary': f'发现{n_issues}个小问题，可在日常中处理',
                'recommended_focus': 'routine_maintenance',
            }

    def get_stats(self):
        total_true = self.accurate_perceptions + self.missed_issues
        return {
            'total_perceptions': self.total_perceptions,
            'detection_rate': round(self.accurate_perceptions / max(total_true, 1), 3),
            'false_alarm_rate': round(self.false_alarms / max(self.total_perceptions, 1), 3),
            'missed_rate': round(self.missed_issues / max(total_true, 1), 3),
        }


# ============================================================
# STATISTICAL PERCEPTION CHANNEL (Non-Linguistic)
# ============================================================
class StatisticalPerceptionChannel:
    """
    Monitors time-series data for slow drift, distributional shifts,
    and periodic anomalies that language-based perception structurally misses.

    This channel has NO semantic understanding. It only sees numbers and
    their temporal patterns. Its strength is exactly where LLM is blind:
    gradual erosion, variance changes, and cycle disruptions.
    """

    def __init__(self, window_size=14):
        self.window = window_size
        self.history = {
            'safety': [], 'compliance': [], 'satisfaction': [],
            'budget_rate': [], 'hidden_risk_count': [],
            'personnel_fatigue_avg': [], 'cert_valid_ratio': [],
            'inspection_interval': [],
        }
        self.baseline = {}
        self.alerts = []
        self.total_alerts = 0
        self.true_positive_alerts = 0

    def update(self, day_result, env_state, day):
        if day_result.get('safety_score') is not None:
            self.history['safety'].append(day_result['safety_score'])
        if day_result.get('compliance_score') is not None:
            self.history['compliance'].append(day_result['compliance_score'])
        self.history['satisfaction'].append(day_result.get('satisfaction', 0.5))
        self.history['hidden_risk_count'].append(day_result.get('hidden_risks', 0))
        self.history['budget_rate'].append(day_result.get('budget_remaining', 0))

        personnel = env_state.get('personnel', {})
        if personnel:
            fatigues = [p['fatigue'] for p in personnel.values()]
            certs = [1.0 if p['health_cert_valid'] else 0.0 for p in personnel.values()]
            self.history['personnel_fatigue_avg'].append(np.mean(fatigues))
            self.history['cert_valid_ratio'].append(np.mean(certs))

        last_insp = env_state.get('compliance', {}).get('last_inspection_day', 0)
        self.history['inspection_interval'].append(day - last_insp)

        for key in self.history:
            if len(self.history[key]) > self.window * 3:
                self.history[key] = self.history[key][-self.window * 3:]

        if day == self.window:
            for key, vals in self.history.items():
                if len(vals) >= self.window:
                    self.baseline[key] = {
                        'mean': float(np.mean(vals[:self.window])),
                        'std': float(np.std(vals[:self.window])) + 1e-6,
                    }

    def perceive(self, day):
        if day < self.window or not self.baseline:
            return {'signals': [], 'alerts': [], 'drift_scores': {}}

        signals = []
        alerts = []
        drift_scores = {}

        for key, vals in self.history.items():
            if len(vals) < self.window or key not in self.baseline:
                continue
            recent = vals[-self.window:]
            base = self.baseline[key]
            recent_mean = float(np.mean(recent))
            drift_z = (recent_mean - base['mean']) / base['std']
            drift_scores[key] = round(drift_z, 2)

            if abs(drift_z) > 1.5:
                direction = 'declining' if drift_z < 0 else 'improving'
                severity = 'critical' if abs(drift_z) > 2.5 else ('high' if abs(drift_z) > 2.0 else 'medium')
                signals.append({
                    'type': 'trend',
                    'value': direction,
                    'dimension': key,
                    'magnitude': round(abs(drift_z), 2),
                    'source': 'statistical_drift',
                    'confidence': min(0.95, 0.5 + abs(drift_z) * 0.15),
                })
                if severity in ('critical', 'high'):
                    alerts.append({
                        'dimension': key,
                        'severity': severity,
                        'drift_z': round(drift_z, 2),
                        'narrative': f'{key} drifted {direction} by {abs(drift_z):.1f}σ from baseline',
                    })
                    self.total_alerts += 1

        recent_intervals = self.history['inspection_interval'][-self.window:]
        if recent_intervals:
            max_gap = max(recent_intervals)
            if max_gap > 14:
                signals.append({
                    'type': 'urgency',
                    'value': 'high' if max_gap > 21 else 'medium',
                    'source': 'statistical_inspection_gap',
                    'confidence': min(0.9, 0.5 + max_gap * 0.02),
                    'dimension': 'inspection_overdue',
                })
                alerts.append({
                    'dimension': 'inspection_interval',
                    'severity': 'high' if max_gap > 21 else 'medium',
                    'drift_z': round(max_gap / 14, 2),
                    'narrative': f'Inspection gap={max_gap} days (threshold=14)',
                })
                self.total_alerts += 1

        recent_risks = self.history['hidden_risk_count'][-self.window:]
        if len(recent_risks) >= self.window:
            risk_trend = np.polyfit(range(len(recent_risks)), recent_risks, 1)[0]
            if risk_trend > 0.1:
                signals.append({
                    'type': 'trend',
                    'value': 'declining',
                    'dimension': 'risk_accumulation',
                    'magnitude': round(risk_trend, 3),
                    'source': 'statistical_risk_trend',
                    'confidence': min(0.85, 0.4 + risk_trend * 2),
                })

        fatigue_recent = self.history['personnel_fatigue_avg'][-self.window:]
        if len(fatigue_recent) >= self.window:
            fatigue_trend = np.polyfit(range(len(fatigue_recent)), fatigue_recent, 1)[0]
            if fatigue_trend > 0.005:
                signals.append({
                    'type': 'trend',
                    'value': 'declining',
                    'dimension': 'personnel_fatigue',
                    'magnitude': round(fatigue_trend, 4),
                    'source': 'statistical_fatigue',
                    'confidence': min(0.8, 0.4 + fatigue_trend * 20),
                })

        cert_recent = self.history['cert_valid_ratio'][-self.window:]
        if len(cert_recent) >= self.window and np.mean(cert_recent) < 0.9:
            signals.append({
                'type': 'domain',
                'value': 'personnel',
                'weight': round(1.0 - np.mean(cert_recent), 2),
                'source': 'statistical_cert_decay',
                'confidence': 0.7,
            })

        self.alerts = alerts
        return {
            'signals': signals,
            'alerts': alerts,
            'drift_scores': drift_scores,
        }

    def get_stats(self):
        return {
            'total_alerts': self.total_alerts,
            'active_alerts': len(self.alerts),
            'monitored_dimensions': len(self.baseline),
        }


# ============================================================
# SHARED ABSTRACTION LAYER
# ============================================================
class SharedAbstractionLayer:
    """
    Translates between perception outputs and TUF-E agent input.
    v0.7: Now handles THREE observers (LLM + Statistical + TUF-E).
    """

    SIGNAL_TYPES = {
        'urgency': ['critical', 'high', 'medium', 'low'],
        'domain': ['inventory', 'personnel', 'equipment', 'production', 'financial'],
        'trend': ['improving', 'stable', 'declining', 'volatile'],
        'action_hint': ['inspect', 'intervene', 'order', 'menu_plan', 'staff_assign', 'record'],
    }

    def translate_perception_to_signals(self, perception):
        signals = []
        assessment = perception.get('overall_assessment', {})
        risk = assessment.get('risk_level', 'low')
        urgency_map = {'high': 'critical', 'medium': 'high', 'low': 'low'}
        signals.append({
            'type': 'urgency',
            'value': urgency_map.get(risk, 'medium'),
            'source': 'llm_assessment',
            'confidence': 0.7,
        })

        domain_counts = defaultdict(int)
        for issue in perception.get('issues', []):
            loc = issue.get('location', '')
            if 'ingredient' in loc or 'storage' in loc:
                domain_counts['inventory'] += 1
            elif any(s in loc for s in ['prep', 'cook', 'serve', 'wash', 'staff']):
                domain_counts['personnel'] += 1
            elif any(e in loc for e in ['fridge', 'freezer', 'stove', 'disinfection']):
                domain_counts['equipment'] += 1
            else:
                domain_counts['production'] += 1

        for domain, count in domain_counts.items():
            signals.append({
                'type': 'domain',
                'value': domain,
                'weight': count,
                'source': 'llm_issues',
                'confidence': min(0.9, 0.4 + count * 0.15),
            })

        focus = assessment.get('recommended_focus', 'routine_maintenance')
        action_map = {
            'urgent_intervention': 'intervene',
            'systematic_inspection': 'inspect',
            'routine_maintenance': 'record',
        }
        hinted_action = action_map.get(focus, 'inspect')
        signals.append({
            'type': 'action_hint',
            'value': hinted_action,
            'source': 'llm_recommendation',
            'confidence': 0.5,
        })

        for trend in perception.get('trend_signals', []):
            signals.append({
                'type': 'trend',
                'value': trend.get('signal', 'stable'),
                'dimension': trend.get('dimension', 'unknown'),
                'source': 'llm_trend',
                'confidence': 0.6,
            })

        return signals

    def compute_three_way_consensus(self, llm_signals, stat_signals, tufe_signals):
        """
        Three-way consensus: LLM vs Statistical vs TUF-E.
        Each pair can agree or disagree; triple disagreement is highest info.
        """
        consensus = {}
        disagreements = []

        all_signal_sets = {
            'llm': llm_signals,
            'stat': stat_signals,
            'tufe': tufe_signals,
        }

        urgency_vals = {}
        for name, sigs in all_signal_sets.items():
            urg = [s for s in sigs if s['type'] == 'urgency']
            if urg:
                urgency_vals[name] = urg[0]['value']

        if len(urgency_vals) >= 2:
            unique_vals = set(urgency_vals.values())
            if len(unique_vals) == 1:
                consensus['urgency'] = {'value': list(unique_vals)[0], 'agreement': 'triple'}
            elif len(unique_vals) == 2:
                pairs = list(urgency_vals.items())
                agreeing = None
                for i, (n1, v1) in enumerate(pairs):
                    for n2, v2 in pairs[i+1:]:
                        if v1 == v2:
                            agreeing = f'{n1}+{n2}'
                outlier = [n for n, v in urgency_vals.items() if v not in [v for n2, v in urgency_vals.items() if n2 != n]]
                disagreements.append({
                    'dimension': 'urgency',
                    'values': urgency_vals,
                    'agreeing_pair': agreeing,
                    'info_content': f'紧急度判断分歧: {urgency_vals}',
                })
            else:
                disagreements.append({
                    'dimension': 'urgency',
                    'values': urgency_vals,
                    'agreeing_pair': None,
                    'info_content': f'三方紧急度全不同: {urgency_vals} — 最高信息量',
                })

        trend_dims = defaultdict(dict)
        for name, sigs in all_signal_sets.items():
            for s in sigs:
                if s['type'] == 'trend' and 'dimension' in s:
                    trend_dims[s['dimension']][name] = s['value']

        for dim, observers in trend_dims.items():
            if len(observers) >= 2:
                vals = set(observers.values())
                if len(vals) == 1:
                    consensus[f'trend_{dim}'] = {'value': list(vals)[0], 'observers': list(observers.keys()), 'agreement': True}
                else:
                    disagreements.append({
                        'dimension': f'trend_{dim}',
                        'values': observers,
                        'info_content': f'{dim}趋势判断分歧: {observers}',
                    })

        domain_sets = {}
        for name, sigs in all_signal_sets.items():
            domain_sets[name] = set(s['value'] for s in sigs if s['type'] == 'domain')

        all_domains = set()
        for ds in domain_sets.values():
            all_domains |= ds

        stat_only = domain_sets.get('stat', set()) - domain_sets.get('llm', set()) - domain_sets.get('tufe', set())
        if stat_only:
            disagreements.append({
                'dimension': 'stat_unique_domain',
                'domains': list(stat_only),
                'info_content': f'统计通道独有关注: {list(stat_only)} — LLM和TUF-E均未注意到的慢变量',
            })

        return {
            'consensus': consensus,
            'disagreements': disagreements,
            'n_agreements': len(consensus),
            'n_disagreements': len(disagreements),
            'information_density': len(disagreements) / max(len(consensus) + len(disagreements), 1),
        }


# ============================================================
# HYBRID AGENT v0.7
# ============================================================
class HybridAgentV07:
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
            'issues_detected_count': 0,
            'successful_interventions': 0,
            'llm_disagreements_seen': 0,
            'consensus_used': 0,
        }
        self.total_actions = 0
        self.successful_actions = 0
        self.energy_bits = 0
        self.recent_outcomes = []

    def choose_action(self, observation, stage_info, llm_signals=None, stat_signals=None, consensus=None):
        available = stage_info['available_actions']
        prefs = {a: max(0.05, self.action_preferences.get(a, 1.0)) for a in available}

        all_perception_signals = (llm_signals or []) + (stat_signals or [])

        if all_perception_signals and consensus:
            for sig in all_perception_signals:
                if sig['type'] == 'action_hint' and sig['value'] in prefs:
                    prefs[sig['value']] *= (1.0 + sig['confidence'] * 0.5)
                if sig['type'] == 'domain':
                    domain_action_map = {
                        'inventory': 'order',
                        'personnel': 'staff_assign',
                        'equipment': 'inspect',
                        'production': 'inspect',
                    }
                    mapped = domain_action_map.get(sig['value'])
                    if mapped and mapped in prefs:
                        prefs[mapped] *= (1.0 + sig.get('weight', 1) * 0.05)
                if sig['type'] == 'trend' and sig.get('value') == 'declining':
                    dim = sig.get('dimension', '')
                    if 'fatigue' in dim or 'cert' in dim or 'personnel' in dim:
                        if 'staff_assign' in prefs:
                            prefs['staff_assign'] *= (1.0 + sig.get('magnitude', 1) * 0.1)
                        if 'inspect' in prefs:
                            prefs['inspect'] *= (1.0 + sig.get('magnitude', 1) * 0.08)
                    elif 'risk' in dim:
                        if 'inspect' in prefs:
                            prefs['inspect'] *= (1.0 + sig.get('magnitude', 1) * 0.12)
                    elif 'inspection' in dim:
                        if 'inspect' in prefs:
                            prefs['inspect'] *= 1.2
                    elif 'budget' in dim:
                        if 'menu_plan' in prefs:
                            prefs['menu_plan'] *= 1.15
                        if 'order' in prefs:
                            prefs['order'] *= 1.1

            if consensus.get('disagreements'):
                for d in consensus['disagreements']:
                    if 'stat_unique' in d.get('dimension', ''):
                        if 'inspect' in prefs:
                            prefs['inspect'] *= 1.15
                        if 'record' in prefs:
                            prefs['record'] *= 1.2
                    elif d.get('dimension') == 'urgency':
                        if 'intervene' in prefs:
                            prefs['intervene'] *= 1.1

        pref_arr = np.array([prefs[a] for a in available])
        pref_arr /= pref_arr.sum()

        exploration = max(0.15, 1.0 - self.self_model['confidence'])
        if self.rng.random() < exploration:
            action = self.rng.choice(available)
        else:
            action = self.rng.choice(available, p=pref_arr)

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

    def learn(self, action, result, day_scores, consensus=None):
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
        if action == 'intervene' and not result['success']:
            reward -= 0.05

        if action == 'menu_plan' and result['success']:
            reward += 0.15
        if action == 'order' and result['success']:
            reward += 0.12
        if action == 'staff_assign' and result['success']:
            reward += 0.12
        if action == 'record' and result['success']:
            reward += 0.08

        if consensus and consensus.get('disagreements'):
            self.self_model['llm_disagreements_seen'] += len(consensus['disagreements'])
            reward += 0.05 * len(consensus['disagreements'])
        if consensus and consensus.get('consensus'):
            self.self_model['consensus_used'] += len(consensus['consensus'])

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

    def generate_tufe_signals(self, observation):
        signals = []
        inv = observation.get('inventory_summary', {})
        if inv.get('low_stock_count', 0) > 3:
            signals.append({'type': 'domain', 'value': 'inventory', 'weight': inv['low_stock_count'], 'source': 'tufe_structural'})
        if inv.get('near_expiry_count', 0) > 1:
            signals.append({'type': 'urgency', 'value': 'high', 'source': 'tufe_structural'})

        pers = observation.get('personnel_summary', {})
        if pers.get('cert_expired_count', 0) > 0:
            signals.append({'type': 'domain', 'value': 'personnel', 'weight': pers['cert_expired_count'], 'source': 'tufe_structural'})

        equip = observation.get('equipment_summary', {})
        if not equip.get('all_functional', True):
            signals.append({'type': 'urgency', 'value': 'critical', 'source': 'tufe_structural'})
            signals.append({'type': 'domain', 'value': 'equipment', 'weight': 1, 'source': 'tufe_structural'})
        if equip.get('maintenance_overdue', 0) > 0:
            signals.append({'type': 'trend', 'value': 'declining', 'dimension': 'equipment', 'source': 'tufe_structural'})

        fin = observation.get('financial', {})
        if fin.get('budget_remaining', 999999) < 80000:
            signals.append({'type': 'trend', 'value': 'declining', 'dimension': 'budget', 'source': 'tufe_structural'})

        if not signals:
            signals.append({'type': 'urgency', 'value': 'low', 'source': 'tufe_structural'})

        return signals

    def get_stats(self):
        return {
            'total_actions': self.total_actions,
            'success_rate': round(self.successful_actions / max(self.total_actions, 1), 3),
            'n_pivots': len(self.pivot_library),
            'confidence': round(self.self_model['confidence'], 3),
            'energy_bits': self.energy_bits,
            'llm_disagreements_seen': self.self_model['llm_disagreements_seen'],
            'consensus_used': self.self_model['consensus_used'],
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
def run_v07_experiment(n_days=120, report_every=10):
    print("=" * 70)
    print("TUF-E Hybrid Agent v0.7: Triple-Observer (LLM + Statistical + TUF-E)")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    env = CafeteriaEnvironment(seed=42, n_days=n_days)
    agent = HybridAgentV07(seed=42)
    llm = LLMPerceptionFunction(seed=42)
    stat = StatisticalPerceptionChannel(window_size=14)
    sal = SharedAbstractionLayer()

    results = {
        'milestones': [], 'final': {}, 'stage_transitions': [],
        'consensus_log': [], 'llm_stats_history': [],
    }

    print(f"\nArchitecture: LLM(感知) + Statistical(统计) → 共享抽象层 ← TUF-E(认知)")
    print(f"Duration: {n_days} days, 4 learning stages")
    print(f"Key question: Does triple-observer consensus outperform dual-observer?")

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
                stage_info = env.get_stage_info()

        raw_data = env.get_raw_data_for_llm()
        true_findings = env.deep_inspect()
        llm_perception = llm.perceive(raw_data, true_findings)
        llm_signals = sal.translate_perception_to_signals(llm_perception)
        tufe_signals = agent.generate_tufe_signals(obs)

        stat_perception = stat.perceive(day + 1)
        stat_signals = stat_perception.get('signals', [])

        consensus = sal.compute_three_way_consensus(llm_signals, stat_signals, tufe_signals)

        actions_per_day = 4 if env.stage == 1 else 6
        day_scores = None

        for _ in range(actions_per_day):
            action, params = agent.choose_action(obs, stage_info, llm_signals, stat_signals, consensus)
            result = env.act(action, params)
            agent.learn(action, result, None, consensus)

        day_scores = env.advance_day()
        stat.update(day_scores, env._true_state, env.day)

        if day_scores:
            agent.recent_outcomes.append(
                day_scores['safety_score'] * 0.3 +
                (day_scores['compliance_score'] if day_scores.get('compliance_score') else 0.5) * 0.3 +
                day_scores['satisfaction'] * 0.2 +
                max(0, day_scores['cost_efficiency']) * 0.2
            )

        if (day + 1) % report_every == 0 and day_scores:
            stats = agent.get_stats()
            llm_stats = llm.get_stats()
            stat_stats = stat.get_stats()
            comply_str = f"{day_scores['compliance_score']:.2f}" if day_scores.get('compliance_score') is not None else "N/A"
            print(f"  Day {day+1:>3}/{n_days} [{stage_info['name'][:4]}]: "
                  f"safety={day_scores['safety_score']:.2f} | "
                  f"comply={comply_str:>5} | "
                  f"sat={day_scores['satisfaction']:.2f} | "
                  f"hidden={day_scores['hidden_risks']} | "
                  f"LLM_det={llm_stats['detection_rate']:.2f} | "
                  f"stat_alerts={stat_stats['active_alerts']} | "
                  f"disagree={consensus['n_disagreements']} | "
                  f"agent_sr={stats['success_rate']:.2f}")

            results['milestones'].append({
                'day': day + 1,
                'stage': stage_info['name'],
                'scores': day_scores,
                'agent_stats': stats,
                'llm_stats': llm_stats,
                'stat_stats': stat_stats,
                'consensus': consensus,
            })
            results['llm_stats_history'].append(llm_stats)

        if consensus['n_disagreements'] > 0:
            results['consensus_log'].append({
                'day': day + 1,
                'disagreements': consensus['disagreements'],
                'info_density': consensus['information_density'],
            })

    final_stats = agent.get_stats()
    final_llm_stats = llm.get_stats()
    final_stat_stats = stat.get_stats()
    s1_end = results['stage_transitions'][0]['day'] if len(results['stage_transitions']) > 0 else 30
    s2_end = results['stage_transitions'][1]['day'] if len(results['stage_transitions']) > 1 else 60
    s3_end = results['stage_transitions'][2]['day'] if len(results['stage_transitions']) > 2 else 90

    results['final'] = {
        'agent_stats': final_stats,
        'llm_stats': final_llm_stats,
        'stat_stats': final_stat_stats,
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
        'total_disagreements': len(results['consensus_log']),
        'avg_info_density': round(float(np.mean([c['info_density'] for c in results['consensus_log']])), 3) if results['consensus_log'] else 0,
    }

    print(f"\n{'=' * 70}")
    print("Final Results — Hybrid v0.7 (Triple Observer)")
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
    print(f"\n  LLM Perception Stats:")
    print(f"    Detection rate: {final_llm_stats['detection_rate']:.3f}")
    print(f"    False alarm rate: {final_llm_stats['false_alarm_rate']:.3f}")
    print(f"    Missed rate: {final_llm_stats['missed_rate']:.3f}")
    print(f"\n  Statistical Channel Stats:")
    print(f"    Total alerts: {final_stat_stats['total_alerts']}")
    print(f"    Active alerts: {final_stat_stats['active_alerts']}")
    print(f"    Monitored dimensions: {final_stat_stats['monitored_dimensions']}")
    print(f"\n  Consensus Dynamics:")
    print(f"    Total disagreement events: {results['final']['total_disagreements']}")
    print(f"    Avg information density: {results['final']['avg_info_density']:.3f}")
    print(f"    Disagreements seen by agent: {final_stats['llm_disagreements_seen']}")
    print(f"    Consensus points used: {final_stats['consensus_used']}")

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
                  f"SR={st['agent_stats_at_transition']['success_rate']:.3f}")

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj

    with open('grounding_v07_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to grounding_v07_results.json")

    return results


if __name__ == '__main__':
    run_v07_experiment(n_days=120, report_every=10)
