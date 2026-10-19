"""
TUF-E Grounded Agent v1.1: Real LLM Integration + Calibrated Simulator

v1.1 基于校准实验结果，将模拟LLM感知替换为真实DashScope API调用。
同时保留校准后的模拟器作为fallback。

校准发现的三个关键参数调整:
  1. SLOW_DRIFT_DETECTION_RATE: 0.3 → 0.6 (真实LLM慢变量检出能力被低估一倍)
  2. 误报机制: 随机8% → 上下文依赖(无问题时0%, 有问题时~27%)
  3. 置信度膨胀确认: 平均81% vs 实际准确率52%

新增:
  - RealLLMPerception: 通过DashScope API调用真实LLM
  - 自动fallback到校准后模拟器(API失败/限流时)
  - 支持 --api-key 和 --model 参数
"""

import numpy as np
import json
import sys
import io
import os
import time
from datetime import datetime
from collections import defaultdict, Counter

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
except Exception:
    pass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# CAFETERIA ENVIRONMENT v1.0 — Lewin-Constrained B=f(P,E)
# ============================================================
class CafeteriaEnvironment:
    """
    v1.0: 引入 supervision_level 作为环境结构参数。
    行为不再无条件成功；合规性有结构性优势；奖励反映真实后果。
    """
    INGREDIENT_CATS = ['meat', 'vegetable', 'seafood', 'grain', 'frozen', 'condiment']
    STATIONS = ['prep', 'cook', 'serve', 'wash']
    N_INGREDIENTS = 12

    def __init__(self, seed=42, n_days=120, supervision_level=0.7):
        self.rng = np.random.default_rng(seed)
        self.n_days = n_days
        self.supervision_level = float(np.clip(supervision_level, 0.0, 1.0))
        self.sop_compliance_cost = 0.5 + 0.5 * self.supervision_level  # 0.5~1.0
        self.day = 0
        self.stage = 1
        self._true_state = self._init_state()
        self._observable_state = self._make_observable(self._true_state)
        self._next_inspection_day = int(self.rng.integers(7, 14))
        self.history = []
        self.daily_scores = {
            'safety': [], 'compliance': [], 'cost': [], 'satisfaction': []
        }
        # v1.0: track menu_plan attempts per day for diminishing returns
        self._menu_plan_count_today = 0
        # v1.0: inspection frequency tracker for risk_exposure decoupling
        self._inspection_count_total = 0
        self._days_since_last_inspection_action = 0

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
            'avg_skill': round(float(np.mean([v['skill_level'] for v in true_state['personnel'].values()])), 2),
            'avg_fatigue': round(float(np.mean([v['fatigue'] for v in true_state['personnel'].values()])), 2),
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

        # v1.0: production findings based on actual state, not pure random
        if area is None or area == 'production':
            # sample_missing: based on personnel compliance in cook/prep stations
            cook_compliance = []
            for pid, p in self._true_state['personnel'].items():
                if p['station'] in ('cook', 'prep'):
                    eff = p.get('_effective_compliance_today', p['compliance_baseline'])
                    cook_compliance.append(eff)
            if cook_compliance:
                avg_cook_comp = float(np.mean(cook_compliance))
                # Lower compliance -> higher chance of sample_missing
                sample_missing_prob = max(0.05, 1.0 - avg_cook_comp)
                if self.rng.random() < sample_missing_prob:
                    findings.append(('sample_missing', 'kitchen', 0.75))

            # cross_contamination: based on equipment functionality and personnel fatigue
            prep_fatigue = []
            for pid, p in self._true_state['personnel'].items():
                if p['station'] == 'prep':
                    prep_fatigue.append(p['fatigue'])
            disinfection_ok = self._true_state['equipment'].get('disinfection_cabinet', {}).get('functional', True)
            cross_contam_base = 0.05
            if prep_fatigue:
                cross_contam_base += float(np.mean(prep_fatigue)) * 0.3
            if not disinfection_ok:
                cross_contam_base += 0.25
            cross_contam_prob = min(0.6, cross_contam_base)
            if self.rng.random() < cross_contam_prob:
                findings.append(('cross_contamination', 'prep_area', 0.65))

        return findings

    def act(self, action_type, params=None):
        result = {'action': action_type, 'success': False, 'feedback': '', 'cost': 0,
                  'actual_impact': {}}

        if action_type == 'inspect':
            area = params.get('area') if params else None
            findings = self.deep_inspect(area)
            result['findings'] = findings
            result['success'] = True
            result['cost'] = 50
            result['feedback'] = f"Found {len(findings)} issues"
            self._inspection_count_total += 1
            self._days_since_last_inspection_action = 0
            for item_type, location, severity in findings:
                self._true_state['compliance']['open_findings'].append({
                    'type': item_type, 'location': location,
                    'severity': severity, 'found_day': self.day
                })
            # v1.0: actual impact — inspections reduce hidden risk accumulation
            result['actual_impact'] = {'findings_count': len(findings), 'risk_prevented': len(findings) * 0.1}

        elif action_type == 'order':
            ingredient_id = params.get('ingredient_id', 0) if params else 0
            quantity = params.get('quantity', 20) if params else 20
            cost = quantity * float(self.rng.uniform(3, 15))
            if cost <= self._true_state['financial']['budget_remaining']:
                self._true_state['financial']['budget_remaining'] -= cost
                old_qty = 0
                if ingredient_id in self._true_state['inventory']:
                    old_qty = self._true_state['inventory'][ingredient_id]['quantity']
                    self._true_state['inventory'][ingredient_id]['quantity'] += quantity
                    self._true_state['inventory'][ingredient_id]['days_remaining'] = int(self.rng.integers(3, 10))
                    verify_rate = 0.7 if self.stage >= 2 else 0.4
                    self._true_state['inventory'][ingredient_id]['received_verified'] = self.rng.random() < verify_rate
                    self._true_state['inventory'][ingredient_id]['has_cert'] = self.rng.random() < 0.85
                result['success'] = True
                result['cost'] = cost
                result['feedback'] = f"Ordered {quantity} units, cost={cost:.0f}"
                # v1.0: actual impact — did we actually need this?
                if old_qty < 10:
                    result['actual_impact'] = {'stock_improved': True, 'waste': 0.0}
                else:
                    # Over-ordering wastes budget
                    result['actual_impact'] = {'stock_improved': False, 'waste': cost * 0.3}
            else:
                result['feedback'] = "Budget exceeded"
                result['actual_impact'] = {'stock_improved': False, 'waste': 0}

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
                    # v1.0: actual impact — severity of what was fixed
                    result['actual_impact'] = {'severity_fixed': finding['severity'],
                                               'type': finding['type']}
                else:
                    result['feedback'] = f"Intervention failed for {finding['type']}"
                    result['actual_impact'] = {'severity_fixed': 0, 'wasted_effort': 0.1}
            else:
                result['feedback'] = "No finding at that index"
                result['actual_impact'] = {'wasted_effort': 0.05}

        elif action_type == 'menu_plan':
            # v1.0: diminishing returns for repeated menu_plan in same day
            self._menu_plan_count_today += 1
            diminishing_factor = 1.0 / (1.0 + 0.5 * (self._menu_plan_count_today - 1))

            budget_per_meal = params.get('budget_per_meal', 12) if params else 12
            taste_weight = params.get('taste_weight', 0.6) if params else 0.6
            nutrition_weight = 1.0 - taste_weight
            base_sat_impact = taste_weight * 0.1 + nutrition_weight * 0.05
            sat_impact = base_sat_impact * diminishing_factor
            cost_impact = budget_per_meal * 300
            if cost_impact <= self._true_state['financial']['budget_remaining']:
                old_sat = self._true_state['satisfaction']['recent_score']
                self._true_state['satisfaction']['recent_score'] = min(1.0,
                    old_sat + sat_impact * float(self.rng.uniform(0.5, 1.5)))
                actual_sat_gain = self._true_state['satisfaction']['recent_score'] - old_sat
                self._true_state['financial']['daily_cost'] = cost_impact
                result['success'] = True
                result['cost'] = cost_impact
                result['feedback'] = f"Menu planned: taste_w={taste_weight:.1f}, nutr_w={nutrition_weight:.1f}, dim={diminishing_factor:.2f}"
                result['actual_impact'] = {'satisfaction_gain': actual_sat_gain,
                                           'cost_incurred': cost_impact,
                                           'diminishing_factor': diminishing_factor}
            else:
                result['feedback'] = "Menu too expensive for remaining budget"
                result['actual_impact'] = {'satisfaction_gain': 0, 'cost_incurred': 0}

        elif action_type == 'staff_assign':
            # v1.0: staff_assign only succeeds when there's actual need
            # Check fatigue, skill match, and supervision
            target_station = params.get('target_station', None) if params else None
            if target_station is None:
                target_station = self.rng.choice(self.STATIONS)

            # Find personnel in target station
            station_staff = [(pid, p) for pid, p in self._true_state['personnel'].items()
                            if p['station'] == target_station]

            if not station_staff:
                result['success'] = False
                result['feedback'] = f"No staff at station {target_station}"
                result['cost'] = 20  # wasted effort
                result['actual_impact'] = {'wasted_effort': 0.1}
                # v1.0: blind reassignment under supervision -> negative feedback
                if self.rng.random() < self.supervision_level:
                    result['feedback'] += " (supervisor flagged unnecessary reassignment)"
                    result['cost'] += 50 * self.supervision_level
                    result['actual_impact']['supervision_penalty'] = True
                return result

            avg_fatigue = float(np.mean([p['fatigue'] for _, p in station_staff]))
            avg_skill = float(np.mean([p['skill_level'] for _, p in station_staff]))
            actual_need = avg_fatigue > 0.5 or avg_skill < 0.6

            if actual_need:
                result['success'] = True
                result['feedback'] = f"Staff reassigned to {target_station} (need: fatigue={avg_fatigue:.2f}, skill={avg_skill:.2f})"
                result['cost'] = 30
                # Actually improve fatigue for one staff member
                if station_staff:
                    target_pid, target_p = station_staff[0]
                    target_p['fatigue'] = max(0.0, target_p['fatigue'] - 0.2)
                result['actual_impact'] = {'fatigue_reduced': 0.2, 'need_met': True}
            else:
                # Unnecessary reassignment
                result['success'] = False
                result['feedback'] = f"Reassignment to {target_station} unnecessary (fatigue={avg_fatigue:.2f}, skill={avg_skill:.2f})"
                result['cost'] = 40  # wasted disruption cost
                result['actual_impact'] = {'wasted_effort': 0.15, 'need_met': False, 'disruption': True}
                # Under supervision, unnecessary actions are penalized
                if self.rng.random() < self.supervision_level * 0.8:
                    penalty = 30 * self.supervision_level
                    result['cost'] += penalty
                    result['feedback'] += f" (supervisor noted unnecessary action, penalty={penalty:.0f})"
                    result['actual_impact']['supervision_penalty'] = True

        elif action_type == 'record':
            # v1.0: record must match actual state. Validate against true state.
            record_accuracy = self._validate_record_accuracy()

            if record_accuracy > 0.7:
                result['success'] = True
                result['feedback'] = f"Records updated (accuracy={record_accuracy:.2f})"
                result['cost'] = 20
                result['actual_impact'] = {'record_accuracy': record_accuracy}
                # Low chance of hidden risk if accurate
                if self.rng.random() < 0.1 * (1.0 - self.supervision_level):
                    self._true_state['hidden_risks'].append({
                        'type': 'record_gap',
                        'day_created': self.day,
                        'exposure_delay': int(self.rng.integers(3, 15))
                    })
            else:
                # Inaccurate record — under supervision, this is caught
                caught_prob = 0.3 + 0.5 * self.supervision_level
                if self.rng.random() < caught_prob:
                    result['success'] = False
                    result['feedback'] = f"Record rejected: accuracy too low ({record_accuracy:.2f}), supervisor flagged"
                    result['cost'] = 60 * self.supervision_level  # correction cost >> execution cost
                    result['actual_impact'] = {'record_accuracy': record_accuracy, 'caught': True}
                    # False record doubles hidden risk exposure probability
                    self._true_state['hidden_risks'].append({
                        'type': 'false_record',
                        'day_created': self.day,
                        'exposure_delay': int(self.rng.integers(2, 8))  # shorter delay = faster exposure
                    })
                else:
                    # Got away with it, but creates hidden risk
                    result['success'] = True
                    result['feedback'] = f"Records updated (accuracy={record_accuracy:.2f}, not caught)"
                    result['cost'] = 20
                    result['actual_impact'] = {'record_accuracy': record_accuracy, 'caught': False}
                    self._true_state['hidden_risks'].append({
                        'type': 'record_gap',
                        'day_created': self.day,
                        'exposure_delay': int(self.rng.integers(3, 15))
                    })
                    # Double the hidden risk exposure probability
                    for risk in self._true_state['hidden_risks']:
                        if risk['day_created'] == self.day:
                            risk['exposure_delay'] = max(1, risk['exposure_delay'] // 2)

        return result

    def _validate_record_accuracy(self):
        """Check how well records match actual state. Returns 0.0-1.0 accuracy."""
        checks = []

        # Check inventory records vs actual
        for inv in self._true_state['inventory'].values():
            # Certificates match?
            if inv['has_cert']:
                checks.append(1.0)
            else:
                checks.append(0.3)  # Missing cert reduces record accuracy
            # Verification status
            if inv['received_verified']:
                checks.append(1.0)
            else:
                checks.append(0.4)

        # Check personnel certs
        for p in self._true_state['personnel'].values():
            if p['health_cert_valid']:
                checks.append(1.0)
            else:
                checks.append(0.2)
            if p['morning_check_passed']:
                checks.append(0.9)
            else:
                checks.append(0.3)

        # Equipment status awareness
        for e in self._true_state['equipment'].values():
            if e['functional']:
                checks.append(1.0)
            else:
                checks.append(0.3)

        if not checks:
            return 0.5
        return float(np.mean(checks))

    def advance_day(self):
        self.day += 1
        self._menu_plan_count_today = 0
        self._days_since_last_inspection_action += 1

        # Inventory decay
        for inv in self._true_state['inventory'].values():
            inv['days_remaining'] -= 1
            if inv['category'] in ('meat', 'seafood', 'frozen'):
                if self.rng.random() < 0.05:
                    inv['temp'] += float(self.rng.uniform(1, 4))
            if inv['days_remaining'] <= 0 and self.rng.random() < 0.3:
                inv['actual_quality'] *= 0.7

        # Personnel dynamics
        for p in self._true_state['personnel'].values():
            p['morning_check_passed'] = self.rng.random() > 0.03
            p['fatigue'] = min(1.0, p['fatigue'] + float(self.rng.uniform(0.02, 0.08)))
            if self.rng.random() < 0.1:
                p['fatigue'] *= 0.5
            inspection_recent = (self.day - self._true_state['compliance']['last_inspection_day']) <= 3
            supervision_boost = 0.2 * p['supervision_sensitivity'] if inspection_recent else 0
            # v1.0: supervision_level also affects daily compliance
            env_supervision_boost = self.supervision_level * p['supervision_sensitivity'] * 0.15
            effective_compliance = max(0.1, min(1.0,
                p['compliance_baseline'] - p['fatigue'] * 0.3 + supervision_boost + env_supervision_boost))
            p['_effective_compliance_today'] = effective_compliance

        # Equipment decay
        for e in self._true_state['equipment'].values():
            if e['functional'] and self.rng.random() < 0.02:
                e['functional'] = False
            e['last_maintenance_days_ago'] += 1

        # Financial
        daily_meals = int(self.rng.integers(280, 350))
        self._true_state['financial']['meals_served'] += daily_meals
        base_cost = daily_meals * float(self.rng.uniform(10, 14))
        self._true_state['financial']['daily_cost'] = base_cost
        self._true_state['financial']['budget_remaining'] -= base_cost

        # v1.0: risk_exposure based on actual hidden_risk count and inspection frequency, NOT stage
        n_hidden = len(self._true_state['hidden_risks'])
        inspection_recency_factor = max(0.1, 1.0 - self._days_since_last_inspection_action * 0.03)
        inspection_freq_factor = min(1.0, self._inspection_count_total / max(self.day, 1))
        # More hidden risks + less inspection = higher exposure chance per risk
        base_exposure = 0.1 + 0.05 * min(n_hidden, 10)
        risk_exposure_chance = min(0.6, base_exposure * (1.0 + (1.0 - inspection_recency_factor) * 0.5))

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

        # Official inspections
        if self.day >= self._next_inspection_day:
            inspection_findings = self.deep_inspect()
            n_findings = len(inspection_findings)
            # v1.0: compliance_score based on actual compliance state, not just finding count
            personnel_compliance_vals = [
                p.get('_effective_compliance_today', p['compliance_baseline'])
                for p in self._true_state['personnel'].values()
            ]
            avg_personnel_compliance = float(np.mean(personnel_compliance_vals)) if personnel_compliance_vals else 0.5
            # Blend: personnel compliance baseline average + finding penalty
            compliance_score = max(0, avg_personnel_compliance - n_findings * 0.05)
            self._true_state['compliance']['last_inspection_day'] = self.day
            self._next_inspection_day = self.day + int(self.rng.integers(7, 21))
            self._inspection_count_total += 1
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

        # v1.0: NO time-based stage thresholds — stage transitions are agent/entropy-driven
        # (removed: if self.day >= stage_thresholds.get(self.stage, 999): self.stage += 1)

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
# SIMULATED LLM PERCEPTION FUNCTION (unchanged from v0.9)
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
            "WARNING {loc}温度异常！当前读数偏高，食品安全受威胁",
            "{loc}冷链可能断裂，温度超标需紧急处理",
            "检测到{loc}温控异常，建议立即排查",
        ],
        'hygiene_lapse': [
            "{loc}人员操作不够规范，卫生习惯需加强",
            "观察到{loc}在操作流程中有疏忽",
            "{loc}的个人卫生执行不到位",
        ],
        'equipment_fault': [
            "ALERT {loc}设备故障！影响正常生产",
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
            "WARNING 加工区域存在交叉污染风险！",
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

    SLOW_DRIFT_DETECTION_RATE = 0.6  # v1.1: calibrated from 0.3 (real LLM detects slow drift at 60%)

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.perception_history = []
        self.total_perceptions = 0
        self.accurate_perceptions = 0
        self.missed_issues = 0
        self.false_alarms = 0

    def perceive(self, raw_data, true_findings):
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

        # v1.1: Context-dependent false positives (calibrated: 0% when clean, ~27% when issues exist)
        if len(true_findings) > 0 and self.rng.random() < 0.27:
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
        # v1.1: calibrated — real LLM shows mean confidence 81% regardless of accuracy
        base = 0.75 + severity * 0.1
        noise = float(self.rng.normal(0, 0.06))
        return round(max(0.4, min(0.99, base + noise)), 2)

    def _extract_trends(self, raw_data):
        signals = []
        budget = raw_data['financial']['budget_remaining']
        if budget < 100000:
            signals.append({
                'dimension': 'budget',
                'signal': 'critical' if budget < 50000 else 'warning',
                'narrative': f"budget remaining {budget:.0f}, {'critical' if budget < 50000 else 'low'}",
            })
        sat = raw_data['satisfaction']['recent_score']
        if sat < 0.6:
            signals.append({
                'dimension': 'satisfaction',
                'signal': 'declining',
                'narrative': f"satisfaction dropped to {sat:.2f}",
            })
        elif sat > 0.9:
            signals.append({
                'dimension': 'satisfaction',
                'signal': 'high',
                'narrative': f"satisfaction good ({sat:.2f})",
            })
        return signals

    def _generate_assessment(self, issues, raw_data):
        n_issues = len(issues)
        if n_issues == 0:
            return {
                'risk_level': 'low',
                'summary': 'no major issues detected',
                'recommended_focus': 'routine_maintenance',
            }
        high_sev = sum(1 for i in issues if i.get('perceived_severity', 0) > 0.7)
        if high_sev >= 2:
            return {
                'risk_level': 'high',
                'summary': f'{high_sev} high-priority issues need immediate action',
                'recommended_focus': 'urgent_intervention',
            }
        elif n_issues >= 3:
            return {
                'risk_level': 'medium',
                'summary': f'{n_issues} issues pending, systematic review recommended',
                'recommended_focus': 'systematic_inspection',
            }
        else:
            return {
                'risk_level': 'low',
                'summary': f'{n_issues} minor issues, handle in routine',
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
# REAL LLM PERCEPTION (DashScope API)
# ============================================================
REAL_LLM_SYSTEM_PROMPT = """你是一个食堂食品安全管理AI助手。分析每日运营数据，识别潜在的食品安全问题和管理风险。

按以下JSON格式输出（不要输出其他内容）：
{
  "issues": [
    {
      "type": "必须是以下之一: equipment_fault, cert_missing, temp_violation, hygiene_lapse, fifo_violation, record_gap, sample_missing, cross_contamination, fatigue_trend, compliance_erosion, inspection_overdue",
      "location": "具体位置或对象名称",
      "severity": 0.0到1.0之间的数字(不是文字),
      "confidence": 0.0到1.0之间的数字(不是文字),
      "narrative": "中文描述"
    }
  ],
  "overall_assessment": {
    "risk_level": "low/medium/high/critical",
    "summary": "整体评估"
  }
}

重要：
- type必须严格使用上面列出的值之一，不要自创类型名
- severity和confidence必须是数字(如0.8)，不是文字(如\"high\")
- 只报告确实观察到的问题。无问题时issues返回空数组"""

LLM_TYPE_MAPPING = {
    'equipment_fault': 'equipment_fault',
    'equipment_temperature': 'temp_violation',
    'equipment_maintenance': 'equipment_fault',
    'cert_missing': 'cert_missing',
    'certificate_expired': 'cert_missing',
    'temp_violation': 'temp_violation',
    'temperature_anomaly': 'temp_violation',
    'inventory_temperature': 'temp_violation',
    'hygiene_lapse': 'hygiene_lapse',
    'hygiene_issue': 'hygiene_lapse',
    'personal_hygiene': 'hygiene_lapse',
    'fifo_violation': 'fifo_violation',
    'inventory_expiring_soon': 'fifo_violation',
    'inventory_rotation': 'fifo_violation',
    'record_gap': 'record_gap',
    'missing_records': 'record_gap',
    'sample_missing': 'sample_missing',
    'cross_contamination': 'cross_contamination',
    'fatigue_trend': 'fatigue_trend',
    'staff_fatigue': 'fatigue_trend',
    'compliance_erosion': 'compliance_erosion',
    'inspection_overdue': 'inspection_overdue',
}

SEVERITY_TEXT_MAP = {
    'critical': 0.95, 'high': 0.8, 'medium': 0.5, 'moderate': 0.5,
    'low': 0.25, 'minor': 0.2, 'none': 0.0,
}


class RealLLMPerception:
    """
    Wraps DashScope API to provide real LLM perception.
    Falls back to calibrated simulator on failure/rate-limit.
    """

    def __init__(self, api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                 model='qwen3.8-27b', fallback_simulator=None):
        self.model = model
        self.fallback = fallback_simulator
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rate_limited = 0
        self.client = None
        if HAS_OPENAI and api_key:
            try:
                self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)
            except Exception:
                self.client = None

    def perceive(self, raw_data, true_findings):
        self.total_calls += 1

        if self.client is None:
            self.failed_calls += 1
            return self.fallback.perceive(raw_data, true_findings)

        observations = raw_data
        user_msg = json.dumps(observations, ensure_ascii=False, indent=2)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REAL_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"分析以下食堂运营数据：\n\n{user_msg}"},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])

            parsed = json.loads(content)
            self.successful_calls += 1

            issues = []
            for item in parsed.get('issues', []):
                raw_type = item.get('type', 'unknown')
                mapped_type = LLM_TYPE_MAPPING.get(raw_type, raw_type)

                raw_sev = item.get('severity', 0.5)
                if isinstance(raw_sev, str):
                    sev = SEVERITY_TEXT_MAP.get(raw_sev.lower(), 0.5)
                else:
                    try:
                        sev = float(raw_sev)
                    except (ValueError, TypeError):
                        sev = 0.5
                sev = max(0.0, min(1.0, sev))

                raw_conf = item.get('confidence', 0.7)
                if isinstance(raw_conf, str):
                    conf = SEVERITY_TEXT_MAP.get(raw_conf.lower(), 0.7)
                else:
                    try:
                        conf = float(raw_conf)
                    except (ValueError, TypeError):
                        conf = 0.7
                conf = max(0.0, min(1.0, conf))

                issues.append({
                    'type': mapped_type,
                    'location': item.get('location', 'unknown'),
                    'narrative': item.get('narrative', ''),
                    'perceived_severity': sev,
                    'true_severity': 0.0,
                    'confidence': conf,
                    'source': 'real_llm',
                    'raw_type': raw_type,
                })

            gt_type_set = {f[0] for f in true_findings}
            tp_count = sum(1 for i in issues if i['type'] in gt_type_set)
            fp_count = len(issues) - tp_count
            missed_types = gt_type_set - {i['type'] for i in issues}
            missed_count = len(missed_types)

            trend_signals = []
            budget = raw_data.get('financial', {}).get('budget_remaining', 500000)
            if budget < 100000:
                trend_signals.append({
                    'dimension': 'budget',
                    'signal': 'critical' if budget < 50000 else 'warning',
                    'narrative': f"budget remaining {budget:.0f}",
                })

            overall = parsed.get('overall_assessment', {})
            perception = {
                'issues': issues,
                'missed': [{'type': t, 'location': ''} for t in missed_types],
                'trend_signals': trend_signals,
                'overall_assessment': {
                    'risk_level': overall.get('risk_level', 'medium'),
                    'summary': overall.get('summary', ''),
                    'recommended_focus': 'urgent_intervention' if overall.get('risk_level') in ('high', 'critical') else 'routine_maintenance',
                },
                'n_detected': tp_count,
                'n_missed': max(0, missed_count),
                'n_false': max(0, fp_count),
                'accuracy': tp_count / max(len(true_findings), 1),
                'source': 'real_llm',
            }
            return perception

        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'rate' in error_str.lower():
                self.rate_limited += 1
                time.sleep(2)
            self.failed_calls += 1
            return self.fallback.perceive(raw_data, true_findings)

    def get_stats(self):
        return {
            'total_calls': self.total_calls,
            'successful': self.successful_calls,
            'failed': self.failed_calls,
            'rate_limited': self.rate_limited,
            'success_rate': round(self.successful_calls / max(self.total_calls, 1), 3),
        }


# ============================================================
# STATISTICAL PERCEPTION CHANNEL (Non-Linguistic)
# ============================================================
class StatisticalPerceptionChannel:
    """
    Monitors time-series data for slow drift, distributional shifts,
    and periodic anomalies that language-based perception structurally misses.
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
            self.history['personnel_fatigue_avg'].append(float(np.mean(fatigues)))
            self.history['cert_valid_ratio'].append(float(np.mean(certs)))

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
                        'narrative': f'{key} drifted {direction} by {abs(drift_z):.1f} sigma from baseline',
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
            risk_trend = float(np.polyfit(range(len(recent_risks)), recent_risks, 1)[0])
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
            fatigue_trend = float(np.polyfit(range(len(fatigue_recent)), fatigue_recent, 1)[0])
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
                'weight': round(1.0 - float(np.mean(cert_recent)), 2),
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
# SELF-EVOLVING SHARED ABSTRACTION LAYER v1.0
# With CCD (Causal Contribution of Disagreement)
# ============================================================
class EvolvingAbstractionLayerV10:
    """
    v1.0: Diversity-protected translation layer with CCD consensus measure.

    Retained from v0.9:
    - Dynamic credit ceiling based on source concentration
    - Diversity tax on dominant sources
    - Exploration bonus for underused signals with positive trend
    - Periodic mean-reversion to prevent lock-in

    New in v1.0:
    - CCD: tracks causal effect of disagreements on subsequent rewards
    - Effective disagreement ratio replaces raw information_density
    - Baseline reward tracking for causal comparison
    """

    SIGNAL_TYPES = {
        'urgency': ['critical', 'high', 'medium', 'low'],
        'domain': ['inventory', 'personnel', 'equipment', 'production', 'financial'],
        'trend': ['improving', 'stable', 'declining', 'volatile'],
        'action_hint': ['inspect', 'intervene', 'order', 'menu_plan', 'staff_assign', 'record'],
    }

    def __init__(self):
        self.signal_credits = defaultdict(lambda: {'credit': 1.0, 'uses': 0, 'good': 0, 'bad': 0})
        self.active_signals_this_turn = []
        self.weight_history = []
        self.total_updates = 0
        self.source_usage_counts = defaultdict(int)
        self.diversity_score_history = []

        # v1.0 CCD tracking
        self.baseline_reward = 0.0
        self.baseline_reward_samples = []
        self.disagreement_history = []  # All disagreements with causal tracking
        self.ccd_history = []  # Per-step CCD values

    def _signal_key(self, sig):
        return (sig.get('source', 'unknown'), sig.get('type', 'unknown'), sig.get('dimension', sig.get('value', 'none')))

    def _get_source(self, sig):
        src = sig.get('source', 'unknown')
        if src.startswith('llm'):
            return 'llm'
        elif src.startswith('stat'):
            return 'stat'
        elif src.startswith('tufe'):
            return 'tufe'
        return src

    def _compute_source_concentration(self):
        total = sum(self.source_usage_counts.values())
        if total < 10:
            return {}
        concentrations = {}
        for src, count in self.source_usage_counts.items():
            concentrations[src] = count / total
        return concentrations

    def _diversity_tax(self, source):
        concentrations = self._compute_source_concentration()
        share = concentrations.get(source, 0)
        if share > 0.6:
            return max(0.3, 1.0 - (share - 0.6) * 1.5)
        return 1.0

    def _exploration_bonus(self, key):
        entry = self.signal_credits[key]
        if entry['uses'] < 5:
            return 1.5
        if entry['uses'] < 15 and entry['good'] > entry['bad']:
            return 1.2
        return 1.0

    def get_credit(self, sig):
        key = self._signal_key(sig)
        entry = self.signal_credits[key]
        if entry['uses'] < 3:
            return 1.0

        raw_credit = entry['credit']
        source = self._get_source(sig)
        tax = self._diversity_tax(source)
        bonus = self._exploration_bonus(key)

        adjusted = raw_credit * tax * bonus
        return max(0.1, min(2.5, adjusted))

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

    def compute_three_way_consensus(self, llm_signals, stat_signals, tufe_signals,
                                     subsequent_reward=None, action_changed=False):
        """
        v1.0 CCD: Causal Contribution of Disagreement.

        The consensus computation retains the original agreement/disagreement
        detection from v0.9, but replaces the information_density metric with
        a causally-grounded measure.
        """
        consensus = {}
        disagreements = []

        all_signal_sets = {
            'llm': llm_signals,
            'stat': stat_signals,
            'tufe': tufe_signals,
        }

        # --- Urgency agreement ---
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
                disagreements.append({
                    'dimension': 'urgency',
                    'values': urgency_vals,
                    'agreeing_pair': agreeing,
                    'info_content': f'urgency disagreement: {urgency_vals}',
                })
            else:
                disagreements.append({
                    'dimension': 'urgency',
                    'values': urgency_vals,
                    'agreeing_pair': None,
                    'info_content': f'three-way urgency disagreement: {urgency_vals}',
                })

        # --- Trend agreement ---
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
                        'info_content': f'{dim} trend disagreement: {observers}',
                    })

        # --- Domain uniqueness ---
        domain_sets = {}
        for name, sigs in all_signal_sets.items():
            domain_sets[name] = set(s['value'] for s in sigs if s['type'] == 'domain')

        stat_only = domain_sets.get('stat', set()) - domain_sets.get('llm', set()) - domain_sets.get('tufe', set())
        if stat_only:
            disagreements.append({
                'dimension': 'stat_unique_domain',
                'domains': list(stat_only),
                'info_content': f'stat-only domains: {list(stat_only)}',
            })

        # v1.0: CCD — Causal Contribution of Disagreement
        # Track causal effect of each disagreement
        if subsequent_reward is not None:
            for d in disagreements:
                d['causal_effect'] = subsequent_reward - self.baseline_reward
                d['action_changed'] = action_changed

        # Update baseline reward (running average of all observed rewards)
        if subsequent_reward is not None:
            self.baseline_reward_samples.append(subsequent_reward)
            if len(self.baseline_reward_samples) <= 20:
                # During initial observation period, just accumulate
                self.baseline_reward = float(np.mean(self.baseline_reward_samples))
            else:
                # Exponential moving average for baseline
                self.baseline_reward = self.baseline_reward * 0.95 + subsequent_reward * 0.05

        # Record disagreements with causal data
        for d in disagreements:
            self.disagreement_history.append(d)

        # Compute CCD: effective disagreements / total disagreements
        effective_disagreements = [d for d in disagreements
                                   if d.get('causal_effect', 0) > 0 and d.get('action_changed', False)]
        ccd = len(effective_disagreements) / max(len(disagreements), 1)
        self.ccd_history.append(ccd)

        return {
            'consensus': consensus,
            'disagreements': disagreements,
            'n_agreements': len(consensus),
            'n_disagreements': len(disagreements),
            # Legacy metric kept for comparison
            'information_density': len(disagreements) / max(len(consensus) + len(disagreements), 1),
            # v1.0 CCD metric
            'ccd': ccd,
            'n_effective_disagreements': len(effective_disagreements),
        }

    def record_active_signals(self, signals):
        self.active_signals_this_turn = [self._signal_key(s) for s in signals]
        for sig in signals:
            source = self._get_source(sig)
            self.source_usage_counts[source] += 1

    def update_credits(self, reward):
        self.total_updates += 1
        is_good = reward > 0.3
        for key in self.active_signals_this_turn:
            entry = self.signal_credits[key]
            entry['uses'] += 1
            if is_good:
                entry['good'] += 1
                entry['credit'] = min(3.0, entry['credit'] * 1.04)
            else:
                entry['bad'] += 1
                entry['credit'] = max(0.1, entry['credit'] * 0.93)

        if self.total_updates % 100 == 0:
            credits = [e['credit'] for e in self.signal_credits.values() if e['uses'] >= 5]
            if credits:
                mean_credit = float(np.mean(credits))
                for key, entry in self.signal_credits.items():
                    if entry['uses'] >= 5:
                        entry['credit'] = entry['credit'] * 0.85 + mean_credit * 0.15

        concentrations = self._compute_source_concentration()
        if concentrations:
            entropy = -sum(p * np.log(p + 1e-10) for p in concentrations.values())
            max_entropy = np.log(max(len(concentrations), 1))
            diversity = entropy / max_entropy if max_entropy > 0 else 1.0
            self.diversity_score_history.append(round(diversity, 3))

        if self.total_updates % 50 == 0:
            snapshot = {}
            for key, entry in self.signal_credits.items():
                if entry['uses'] >= 5:
                    snapshot[str(key)] = round(entry['credit'], 3)
            self.weight_history.append(snapshot)

    def get_evolution_stats(self):
        stats = {
            'total_signal_types': len(self.signal_credits),
            'total_updates': self.total_updates,
            'evolved_signals': {},
            'source_concentration': self._compute_source_concentration(),
            'diversity_score': round(float(np.mean(self.diversity_score_history[-10:])), 3) if self.diversity_score_history else None,
            'ccd_stats': {
                'avg_ccd': round(float(np.mean(self.ccd_history[-20:])), 3) if self.ccd_history else 0,
                'total_disagreements_tracked': len(self.disagreement_history),
                'baseline_reward': round(self.baseline_reward, 4),
            },
        }
        for key, entry in self.signal_credits.items():
            if entry['uses'] >= 5:
                stats['evolved_signals'][str(key)] = {
                    'credit': round(entry['credit'], 3),
                    'uses': entry['uses'],
                    'good_rate': round(entry['good'] / max(entry['uses'], 1), 3),
                }
        top = sorted(stats['evolved_signals'].items(), key=lambda x: x[1]['credit'], reverse=True)
        stats['top_credited'] = top[:5]
        bottom = sorted(stats['evolved_signals'].items(), key=lambda x: x[1]['credit'])
        stats['lowest_credited'] = bottom[:5]
        return stats


# ============================================================
# HYBRID AGENT v1.0 — Entropy + Bayesian Stage Transitions
# ============================================================
class HybridAgentV10:
    """
    v1.0 agent with:
    - Entropy-based action monitoring
    - Bayesian model saturation tracking
    - Reward-based learning reflecting actual environmental consequences
    - Stage transitions driven by entropy/saturation, not hardcoded thresholds
    """
    OP_NAMES = {
        'inspect': 'inspection/acceptance',
        'order': 'procurement',
        'menu_plan': 'menu planning',
        'intervene': 'field intervention',
        'staff_assign': 'staff reassignment',
        'record': 'record keeping',
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
            'finding_types_seen': set(),
        }
        self.total_actions = 0
        self.successful_actions = 0
        self.energy_bits = 0
        self.recent_outcomes = []

        # v1.0: Entropy tracking
        self.recent_actions = []  # Last N action choices for entropy
        self.action_entropy_history = []

        # v1.0: Model saturation tracking (Bayesian posterior update magnitudes)
        self.pivot_update_magnitudes = []
        self.model_saturation_history = []

        # v1.0: Intervention success tracking for stage transition
        self.intervention_success_window = []  # (success: bool) per intervention

    def compute_action_entropy(self, recent_actions_window=20):
        """Shannon entropy of action choice distribution. Low = certain (possibly overfit), High = exploring."""
        if len(self.recent_actions) < recent_actions_window:
            window = self.recent_actions
        else:
            window = self.recent_actions[-recent_actions_window:]
        if not window:
            return 0.0
        counts = Counter(window)
        n = len(window)
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)

    def compute_model_saturation(self):
        """Mean of recent pivot_library update magnitudes. Approaching 0 means knowledge saturation."""
        if len(self.pivot_update_magnitudes) < 10:
            return 0.0
        recent = self.pivot_update_magnitudes[-10:]
        return float(np.mean(recent))

    def choose_action(self, observation, stage_info, llm_signals=None, stat_signals=None, consensus=None, sal=None):
        available = stage_info['available_actions']
        prefs = {a: max(0.05, self.action_preferences.get(a, 1.0)) for a in available}

        all_perception_signals = (llm_signals or []) + (stat_signals or [])

        if all_perception_signals and consensus:
            for sig in all_perception_signals:
                credit = sal.get_credit(sig) if sal else 1.0
                if sig['type'] == 'action_hint' and sig['value'] in prefs:
                    prefs[sig['value']] *= (1.0 + sig['confidence'] * 0.5 * credit)
                if sig['type'] == 'domain':
                    domain_action_map = {
                        'inventory': 'order',
                        'personnel': 'staff_assign',
                        'equipment': 'inspect',
                        'production': 'inspect',
                    }
                    mapped = domain_action_map.get(sig['value'])
                    if mapped and mapped in prefs:
                        prefs[mapped] *= (1.0 + sig.get('weight', 1) * 0.05 * credit)
                if sig['type'] == 'trend' and sig.get('value') == 'declining':
                    dim = sig.get('dimension', '')
                    mag = sig.get('magnitude', 1) * credit
                    if 'fatigue' in dim or 'cert' in dim or 'personnel' in dim:
                        if 'staff_assign' in prefs:
                            prefs['staff_assign'] *= (1.0 + mag * 0.1)
                        if 'inspect' in prefs:
                            prefs['inspect'] *= (1.0 + mag * 0.08)
                    elif 'risk' in dim:
                        if 'inspect' in prefs:
                            prefs['inspect'] *= (1.0 + mag * 0.12)
                    elif 'inspection' in dim:
                        if 'inspect' in prefs:
                            prefs['inspect'] *= (1.0 + 0.2 * credit)
                    elif 'budget' in dim:
                        if 'menu_plan' in prefs:
                            prefs['menu_plan'] *= (1.0 + 0.15 * credit)
                        if 'order' in prefs:
                            prefs['order'] *= (1.0 + 0.1 * credit)

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

        # Track action for entropy computation
        self.recent_actions.append(action)

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
        elif action == 'staff_assign':
            stations = ['prep', 'cook', 'serve', 'wash']
            return {'target_station': self.rng.choice(stations)}
        return {}

    def learn(self, action, result, day_scores, consensus=None):
        """
        v1.0: Reward is based on actual environmental consequences, not fixed values.
        Positive reward = actually improved a metric.
        Negative reward = actually worsened a metric or wasted resources.
        """
        self.total_actions += 1
        self.energy_bits += 256

        reward = 0.0
        impact = result.get('actual_impact', {})

        if result['success']:
            self.successful_actions += 1

        # --- Reward from actual impacts ---
        if action == 'inspect':
            findings = result.get('findings', [])
            if findings:
                new_findings = len(findings)
                self.self_model['known_issues'] = findings[:3]
                self.self_model['issues_detected_count'] += new_findings
                for f in findings:
                    self.self_model['finding_types_seen'].add(f[0])
                # Diminishing returns on detection
                diminishing = max(0.02, 0.15 / (1 + self.self_model['issues_detected_count'] / 50))
                reward += diminishing
                # Actual impact: prevented risks
                risk_prevented = impact.get('risk_prevented', 0)
                reward += risk_prevented * 0.5
            else:
                # Inspection found nothing — could be good (clean) or bad (missed)
                reward += 0.02  # small positive for verifying

        elif action == 'intervene':
            if result['success']:
                self.self_model['successful_interventions'] += 1
                self.intervention_success_window.append(True)
                severity = impact.get('severity_fixed', 0.5)
                # Higher severity fixed = more reward (prevented bigger problem)
                reward += 0.1 + severity * 0.3
            else:
                self.intervention_success_window.append(False)
                wasted = impact.get('wasted_effort', 0.05)
                reward -= wasted

        elif action == 'order':
            if result['success']:
                stock_improved = impact.get('stock_improved', False)
                waste = impact.get('waste', 0)
                if stock_improved:
                    reward += 0.15  # Actually needed
                else:
                    reward -= waste / 1000.0  # Wasted budget penalty
            else:
                reward -= 0.05

        elif action == 'menu_plan':
            if result['success']:
                sat_gain = impact.get('satisfaction_gain', 0)
                cost_incurred = impact.get('cost_incurred', 0)
                # Reward proportional to actual satisfaction gain
                reward += sat_gain * 2.0
                # Cost penalty for expensive plans
                if cost_incurred > 4000:
                    reward -= 0.05
            else:
                reward -= 0.03

        elif action == 'staff_assign':
            if result['success']:
                need_met = impact.get('need_met', False)
                fatigue_reduced = impact.get('fatigue_reduced', 0)
                reward += fatigue_reduced * 0.5
                if need_met:
                    reward += 0.1
            else:
                wasted = impact.get('wasted_effort', 0)
                reward -= wasted
                if impact.get('supervision_penalty', False):
                    reward -= 0.15  # Significant penalty for unnecessary reassignment

        elif action == 'record':
            if result['success']:
                accuracy = impact.get('record_accuracy', 0.5)
                reward += accuracy * 0.1  # Accurate records are mildly rewarded
                if impact.get('caught', False):
                    reward -= 0.2  # Caught making bad record
            else:
                reward -= 0.15  # Failed record attempt
                if impact.get('caught', False):
                    reward -= 0.2  # Caught by supervisor

        # Consensus impact
        if consensus and consensus.get('disagreements'):
            self.self_model['llm_disagreements_seen'] += len(consensus['disagreements'])
            # v1.0: Only reward disagreements that have causal effect (CCD)
            ccd = consensus.get('ccd', 0)
            reward += 0.03 * ccd * len(consensus['disagreements'])
        if consensus and consensus.get('consensus'):
            self.self_model['consensus_used'] += len(consensus['consensus'])

        # Day-level scores contribution
        if day_scores:
            reward += day_scores.get('safety_score', 0.5) * 0.2
            if day_scores.get('compliance_score') is not None:
                reward += day_scores['compliance_score'] * 0.2
            reward += day_scores.get('satisfaction', 0.5) * 0.1

        # Update action preferences based on actual reward
        self.action_preferences[action] = max(0.05, self.action_preferences.get(action, 1.0) * (1.0 + 0.08 * reward))
        total = sum(self.action_preferences.values())
        for k in self.action_preferences:
            self.action_preferences[k] /= total

        # Update pivot library with magnitude tracking for saturation
        pivot_key = (action, result['success'], 'good' if reward > 0.3 else 'poor')
        if pivot_key in self.pivot_library:
            old_avg = self.pivot_library[pivot_key]['avg_reward']
            self.pivot_library[pivot_key]['count'] += 1
            self.pivot_library[pivot_key]['avg_reward'] = old_avg * 0.9 + reward * 0.1
            # Track update magnitude for saturation
            magnitude = abs(self.pivot_library[pivot_key]['avg_reward'] - old_avg)
            self.pivot_update_magnitudes.append(magnitude)
        else:
            self.pivot_library[pivot_key] = {'count': 1, 'avg_reward': reward}
            self.pivot_update_magnitudes.append(abs(reward))  # New entry = full magnitude

        self.recent_outcomes.append(reward)
        if len(self.recent_outcomes) >= 10:
            recent_avg = float(np.mean(self.recent_outcomes[-20:]))
            self.self_model['confidence'] = min(0.9, max(0.1, 0.3 + recent_avg))

        # Record entropy and saturation after each action
        entropy = self.compute_action_entropy()
        self.action_entropy_history.append(entropy)
        saturation = self.compute_model_saturation()
        self.model_saturation_history.append(saturation)

    def ready_for_stage(self, target_stage):
        """
        v1.0: Entropy + Bayesian stage transitions.
        Replaces hardcoded thresholds with entropy/saturation-based conditions.
        """
        entropy = self.compute_action_entropy()
        saturation = self.compute_model_saturation()

        if target_stage == 2:
            # S1->S2: Action entropy has dropped from initial high (formed preferences)
            # AND model saturation > threshold (enough observations)
            if len(self.action_entropy_history) < 20:
                return False
            initial_entropy = float(np.mean(self.action_entropy_history[:10])) if len(self.action_entropy_history) >= 10 else 2.0
            entropy_drop_ratio = entropy / max(initial_entropy, 0.01)
            # Entropy should have dropped to <80% of initial
            entropy_condition = entropy_drop_ratio < 0.8
            # Should have enough observation data
            saturation_condition = self.self_model['issues_detected_count'] >= 15
            return entropy_condition and saturation_condition

        elif target_stage == 3:
            # S2->S3: Intervention success rate stable (sliding window variance < threshold)
            # AND enough different finding types discovered
            if len(self.intervention_success_window) < 10:
                return False
            recent_window = self.intervention_success_window[-20:]
            if len(recent_window) < 10:
                return False
            success_rate = sum(1 for s in recent_window if s) / len(recent_window)
            # Variance of binary = p*(1-p), should be low (stable success rate)
            variance = success_rate * (1.0 - success_rate)
            variance_condition = variance < 0.2  # Stable success rate
            # Enough different finding types
            finding_diversity = len(self.self_model.get('finding_types_seen', set()))
            diversity_condition = finding_diversity >= 5
            return variance_condition and diversity_condition

        elif target_stage == 4:
            # S3->S4: Model saturation high (rules internalized) AND
            # action entropy abnormally low (habituation beginning)
            saturation_condition = saturation < 0.01  # Very small updates = knowledge saturated
            if len(self.action_entropy_history) < 30:
                return False
            recent_entropy = float(np.mean(self.action_entropy_history[-10:]))
            # Entropy abnormally low compared to mid-history
            mid_start = len(self.action_entropy_history) // 3
            mid_end = 2 * len(self.action_entropy_history) // 3
            if mid_end > mid_start:
                mid_entropy = float(np.mean(self.action_entropy_history[mid_start:mid_end]))
            else:
                mid_entropy = 2.0
            entropy_condition = recent_entropy < mid_entropy * 0.6
            return saturation_condition and entropy_condition

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
        entropy = self.compute_action_entropy()
        saturation = self.compute_model_saturation()
        return {
            'total_actions': self.total_actions,
            'success_rate': round(self.successful_actions / max(self.total_actions, 1), 3),
            'n_pivots': len(self.pivot_library),
            'confidence': round(self.self_model['confidence'], 3),
            'energy_bits': self.energy_bits,
            'llm_disagreements_seen': self.self_model['llm_disagreements_seen'],
            'consensus_used': self.self_model['consensus_used'],
            'action_entropy': round(entropy, 4),
            'model_saturation': round(saturation, 6),
            'finding_types_seen': len(self.self_model.get('finding_types_seen', set())),
            'intervention_success_rate': (
                round(sum(1 for s in self.intervention_success_window if s) /
                      max(len(self.intervention_success_window), 1), 3)
            ),
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
# EXPERIMENT v1.0
# ============================================================
def run_v10_experiment(n_days=120, report_every=10, seed=42, supervision_level=0.7,
                       api_key=None, model='qwen3.8-27b', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'):
    use_real_llm = api_key is not None and HAS_OPENAI
    mode_str = "REAL LLM" if use_real_llm else "CALIBRATED SIMULATOR"
    print("=" * 70)
    print(f"TUF-E Hybrid Agent v1.1: {mode_str}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Seed: {seed}, Supervision level: {supervision_level}")
    if use_real_llm:
        print(f"Model: {model}, Base URL: {base_url}")
    else:
        print("Mode: Calibrated simulator (slow_drift=0.6, context-dependent FP, confidence=0.81)")
    print("=" * 70)

    env = CafeteriaEnvironment(seed=seed, n_days=n_days, supervision_level=supervision_level)
    agent = HybridAgentV10(seed=seed)
    llm_sim = LLMPerceptionFunction(seed=seed)  # calibrated simulator as fallback
    if use_real_llm:
        llm = RealLLMPerception(api_key=api_key, base_url=base_url, model=model,
                                fallback_simulator=llm_sim)
    else:
        llm = llm_sim
    stat = StatisticalPerceptionChannel(window_size=14)
    sal = EvolvingAbstractionLayerV10()

    results = {
        'milestones': [], 'final': {}, 'stage_transitions': [],
        'consensus_log': [], 'llm_stats_history': [],
        'action_entropy_history': [],
        'model_saturation_history': [],
        'ccd_history': [],
        'supervision_level': supervision_level,
        'seed': seed,
    }

    print(f"\nArchitecture: LLM + Statistical -> CCD-Enabled SAL <- TUF-E")
    print(f"Duration: {n_days} days, entropy-driven stage transitions")
    print(f"Key questions:")
    print(f"  1. Does supervision_level={supervision_level} create structural compliance advantage?")
    print(f"  2. Does CCD filter out causally inert disagreements?")
    print(f"  3. Do entropy/saturation transitions align with genuine learning?")

    # Track previous action for CCD action_changed detection
    prev_action = None

    for day in range(n_days):
        obs = env.observe()
        stage_info = env.get_stage_info()

        # v1.0: Entropy-driven stage transitions
        if agent.ready_for_stage(env.stage + 1) and env.stage < 4:
            transition = env.try_advance_stage(True)
            if transition:
                old_s, new_s = transition
                entropy = agent.compute_action_entropy()
                saturation = agent.compute_model_saturation()
                print(f"\n  >>> STAGE TRANSITION at day {day+1}: Stage {old_s}->{new_s} ({env.get_stage_info()['name']})")
                print(f"      Entropy: {entropy:.3f}, Saturation: {saturation:.5f}")
                print(f"      Issues: {agent.self_model['issues_detected_count']}, "
                      f"Interventions: {agent.self_model['successful_interventions']}, "
                      f"Finding types: {len(agent.self_model.get('finding_types_seen', set()))}")
                results['stage_transitions'].append({
                    'day': day + 1,
                    'from_stage': old_s,
                    'to_stage': new_s,
                    'entropy_at_transition': round(entropy, 4),
                    'saturation_at_transition': round(saturation, 6),
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

        # v1.0: Compute consensus with CCD tracking
        # Use previous day's reward as subsequent_reward for yesterday's disagreements
        prev_reward = agent.recent_outcomes[-1] if agent.recent_outcomes else None

        # Determine if action changed based on consensus
        # (compare what agent would have done without disagreement vs with)
        current_action_pref = max(agent.action_preferences, key=agent.action_preferences.get)

        consensus = sal.compute_three_way_consensus(
            llm_signals, stat_signals, tufe_signals,
            subsequent_reward=prev_reward,
            action_changed=(prev_action is not None and current_action_pref != prev_action)
        )

        actions_per_day = 4 if env.stage == 1 else 6
        day_scores = None

        all_perception_signals = llm_signals + stat_signals

        day_first_action = None
        for action_idx in range(actions_per_day):
            action, params = agent.choose_action(obs, stage_info, llm_signals, stat_signals, consensus, sal)
            if action_idx == 0:
                day_first_action = action
            result = env.act(action, params)
            sal.record_active_signals(all_perception_signals)
            agent.learn(action, result, None, consensus)
            last_reward = agent.recent_outcomes[-1] if agent.recent_outcomes else 0
            sal.update_credits(last_reward)

        day_scores = env.advance_day()
        stat.update(day_scores, env._true_state, env.day)

        if day_scores:
            composite_reward = (
                day_scores['safety_score'] * 0.3 +
                (day_scores['compliance_score'] if day_scores.get('compliance_score') else 0.5) * 0.3 +
                day_scores['satisfaction'] * 0.2 +
                max(0, day_scores['cost_efficiency']) * 0.2
            )
            agent.recent_outcomes.append(composite_reward)

        # Track metrics per day
        entropy = agent.compute_action_entropy()
        saturation = agent.compute_model_saturation()
        results['action_entropy_history'].append(round(entropy, 4))
        results['model_saturation_history'].append(round(saturation, 6))
        results['ccd_history'].append(consensus.get('ccd', 0))

        prev_action = day_first_action

        if (day + 1) % report_every == 0 and day_scores:
            stats = agent.get_stats()
            llm_stats = llm.get_stats() if not use_real_llm else {
                **llm_sim.get_stats(),
                'real_llm': llm.get_stats(),
            }
            stat_stats = stat.get_stats()
            comply_str = f"{day_scores['compliance_score']:.2f}" if day_scores.get('compliance_score') is not None else "N/A"
            print(f"  Day {day+1:>3}/{n_days} [{stage_info['name'][:4]}]: "
                  f"safety={day_scores['safety_score']:.2f} | "
                  f"comply={comply_str:>5} | "
                  f"sat={day_scores['satisfaction']:.2f} | "
                  f"hidden={day_scores['hidden_risks']} | "
                  f"LLM_det={llm_stats['detection_rate']:.2f} | "
                  f"stat_alerts={stat_stats['active_alerts']} | "
                  f"CCD={consensus.get('ccd', 0):.2f} | "
                  f"H={entropy:.2f} | "
                  f"sat={saturation:.4f} | "
                  f"agent_sr={stats['success_rate']:.2f}")

            results['milestones'].append({
                'day': day + 1,
                'stage': stage_info['name'],
                'scores': day_scores,
                'agent_stats': stats,
                'llm_stats': llm_stats,
                'stat_stats': stat_stats,
                'consensus': consensus,
                'action_entropy': round(entropy, 4),
                'model_saturation': round(saturation, 6),
            })
            results['llm_stats_history'].append(llm_stats)

        if consensus['n_disagreements'] > 0:
            results['consensus_log'].append({
                'day': day + 1,
                'disagreements': consensus['disagreements'],
                'information_density': consensus['information_density'],
                'ccd': consensus['ccd'],
                'n_effective_disagreements': consensus['n_effective_disagreements'],
            })

    # --- Final Results ---
    final_stats = agent.get_stats()
    final_llm_stats = llm.get_stats() if not use_real_llm else {
        **llm_sim.get_stats(),
        'real_llm': llm.get_stats(),
    }
    final_stat_stats = stat.get_stats()

    # Compute stage boundaries from actual transitions
    transitions = results['stage_transitions']
    s1_end = transitions[0]['day'] if len(transitions) > 0 else n_days
    s2_end = transitions[1]['day'] if len(transitions) > 1 else n_days
    s3_end = transitions[2]['day'] if len(transitions) > 2 else n_days

    def safe_mean(arr):
        return round(float(np.mean(arr)), 3) if arr else None

    results['final'] = {
        'agent_stats': final_stats,
        'llm_stats': final_llm_stats,
        'stat_stats': final_stat_stats,
        'score_trends': {
            'avg_safety_stage1': safe_mean(env.daily_scores['safety'][:s1_end]),
            'avg_safety_stage2': safe_mean(env.daily_scores['safety'][s1_end:s2_end]),
            'avg_safety_stage3': safe_mean(env.daily_scores['safety'][s2_end:s3_end]),
            'avg_safety_stage4': safe_mean(env.daily_scores['safety'][s3_end:]) if s3_end < n_days else None,
            'avg_satisfaction_stage1': safe_mean(env.daily_scores['satisfaction'][:s1_end]),
            'avg_satisfaction_stage2': safe_mean(env.daily_scores['satisfaction'][s1_end:s2_end]),
            'avg_satisfaction_stage3': safe_mean(env.daily_scores['satisfaction'][s2_end:s3_end]),
            'avg_satisfaction_stage4': safe_mean(env.daily_scores['satisfaction'][s3_end:]) if s3_end < n_days else None,
            'avg_compliance_stage1': safe_mean(env.daily_scores['compliance'][:s1_end]),
            'avg_compliance_stage2': safe_mean(env.daily_scores['compliance'][s1_end:s2_end]),
            'avg_compliance_stage3': safe_mean(env.daily_scores['compliance'][s2_end:s3_end]),
            'avg_compliance_stage4': safe_mean(env.daily_scores['compliance'][s3_end:]) if s3_end < n_days else None,
        },
        'final_budget': round(env._true_state['financial']['budget_remaining'], 0),
        'total_disagreements': len(results['consensus_log']),
        'avg_info_density': round(float(np.mean([c['information_density'] for c in results['consensus_log']])), 3) if results['consensus_log'] else 0,
        'avg_ccd': round(float(np.mean([c['ccd'] for c in results['consensus_log']])), 3) if results['consensus_log'] else 0,
        'total_effective_disagreements': sum(c.get('n_effective_disagreements', 0) for c in results['consensus_log']),
        'action_entropy_final': results['action_entropy_history'][-1] if results['action_entropy_history'] else None,
        'model_saturation_final': results['model_saturation_history'][-1] if results['model_saturation_history'] else None,
        'supervision_level': supervision_level,
        'seed': seed,
    }

    # --- Console Report ---
    print(f"\n{'=' * 70}")
    print(f"Final Results -- Hybrid v1.0 (Lewin-Constrained, CCD, Entropy-Driven)")
    print(f"{'=' * 70}")
    ts = results['final']['score_trends']

    def fmt(val):
        return f"{val:.3f}" if val is not None else "N/A"

    print(f"  Safety trend:       S1={fmt(ts['avg_safety_stage1'])} -> S2={fmt(ts['avg_safety_stage2'])} -> S3={fmt(ts['avg_safety_stage3'])} -> S4={fmt(ts['avg_safety_stage4'])}")
    print(f"  Satisfaction trend: S1={fmt(ts['avg_satisfaction_stage1'])} -> S2={fmt(ts['avg_satisfaction_stage2'])} -> S3={fmt(ts['avg_satisfaction_stage3'])} -> S4={fmt(ts['avg_satisfaction_stage4'])}")
    print(f"  Compliance trend:   S1={fmt(ts['avg_compliance_stage1'])} -> S2={fmt(ts['avg_compliance_stage2'])} -> S3={fmt(ts['avg_compliance_stage3'])} -> S4={fmt(ts['avg_compliance_stage4'])}")
    print(f"  Final budget: {results['final']['final_budget']:.0f}")
    print(f"  Agent success rate: {final_stats['success_rate']:.3f}")
    print(f"  Pivots learned: {final_stats['n_pivots']}")
    print(f"  Confidence: {final_stats['confidence']:.3f}")

    print(f"\n  [Redesign 1] Lewin Constraint B=f(P,E):")
    print(f"    Supervision level: {supervision_level}")
    print(f"    SOP compliance cost: {env.sop_compliance_cost:.2f}")
    print(f"    Intervention success rate: {final_stats['intervention_success_rate']:.3f}")
    print(f"    Finding types discovered: {final_stats['finding_types_seen']}")

    print(f"\n  [Redesign 2] CCD (Causal Contribution of Disagreement):")
    print(f"    Total disagreement events: {results['final']['total_disagreements']}")
    print(f"    Avg information_density (legacy): {results['final']['avg_info_density']:.3f}")
    print(f"    Avg CCD (new): {results['final']['avg_ccd']:.3f}")
    print(f"    Total effective disagreements: {results['final']['total_effective_disagreements']}")
    print(f"    Disagreements seen by agent: {final_stats['llm_disagreements_seen']}")
    print(f"    Consensus points used: {final_stats['consensus_used']}")

    print(f"\n  [Redesign 3] Entropy + Bayesian Stage Transitions:")
    print(f"    Final action entropy: {final_stats['action_entropy']:.4f}")
    print(f"    Final model saturation: {final_stats['model_saturation']:.6f}")
    if results['stage_transitions']:
        for st in results['stage_transitions']:
            print(f"    Day {st['day']}: Stage {st['from_stage']}->{st['to_stage']}, "
                  f"entropy={st['entropy_at_transition']:.3f}, sat={st['saturation_at_transition']:.5f}")
    else:
        print(f"    No stage transitions occurred")

    print(f"\n  LLM Perception Stats:")
    print(f"    Detection rate: {final_llm_stats['detection_rate']:.3f}")
    print(f"    False alarm rate: {final_llm_stats['false_alarm_rate']:.3f}")
    print(f"    Missed rate: {final_llm_stats['missed_rate']:.3f}")

    print(f"\n  Statistical Channel Stats:")
    print(f"    Total alerts: {final_stat_stats['total_alerts']}")
    print(f"    Active alerts: {final_stat_stats['active_alerts']}")
    print(f"    Monitored dimensions: {final_stat_stats['monitored_dimensions']}")

    evo_stats = sal.get_evolution_stats()
    print(f"\n  Abstraction Layer Evolution:")
    print(f"    Signal types tracked: {evo_stats['total_signal_types']}")
    print(f"    Total credit updates: {evo_stats['total_updates']}")
    print(f"    Diversity score: {evo_stats.get('diversity_score', 'N/A')}")
    print(f"    Source concentration: {evo_stats.get('source_concentration', {})}")
    print(f"    CCD stats: {evo_stats.get('ccd_stats', {})}")
    print(f"    Top credited signals:")
    for name, data in evo_stats['top_credited']:
        print(f"      {name}: credit={data['credit']:.3f}, uses={data['uses']}, good_rate={data['good_rate']:.3f}")
    print(f"    Lowest credited signals:")
    for name, data in evo_stats['lowest_credited']:
        print(f"      {name}: credit={data['credit']:.3f}, uses={data['uses']}, good_rate={data['good_rate']:.3f}")

    print(f"\n  Action preferences:")
    for action, pref in final_stats['top_preferences']:
        print(f"    {action}: {pref:.3f}")

    print(f"\n  Top pivots:")
    for p in final_stats['top_pivots'][:5]:
        success_mark = 'Y' if p['success'] else 'N'
        print(f"    {p['action']}({success_mark},{p['quality']}): count={p['count']}, reward={p['avg_reward']:.3f}")

    # --- Save results ---
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        if isinstance(obj, set): return list(obj)
        return obj

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'grounding_v10_results_seed{seed}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


# ============================================================
# MAIN
# ============================================================
def main(seed=42, n_days=120, supervision_level=0.7, report_every=10,
         api_key=None, model='qwen3.8-27b', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'):
    """
    Run TUF-E v1.1 experiment with optional real LLM integration.
    """
    return run_v10_experiment(
        n_days=n_days,
        report_every=report_every,
        seed=seed,
        supervision_level=supervision_level,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TUF-E Grounded Agent v1.1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--days', type=int, default=120, help='Number of simulation days')
    parser.add_argument('--supervision', type=float, default=0.7, help='Supervision level (0.0-1.0)')
    parser.add_argument('--report-every', type=int, default=10, help='Report interval in days')
    parser.add_argument('--api-key', type=str, default=None, help='DashScope API key for real LLM')
    parser.add_argument('--model', type=str, default='qwen3.8-27b', help='Model name')
    parser.add_argument('--base-url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                       help='API base URL')
    args = parser.parse_args()

    main(
        seed=args.seed,
        n_days=args.days,
        supervision_level=args.supervision,
        report_every=args.report_every,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )
