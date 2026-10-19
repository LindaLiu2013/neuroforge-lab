"""
TUF-E Real LLM Calibration Experiment

Purpose: Compare real LLM perception behavior against the simulator's assumptions.
Tests 6 key hypotheses:
  H1: Salience bias (dramatic issues detected more than slow drift)
  H2: Slow drift detection rate ~30%
  H3: False positive rate ~8%
  H4: Confidence miscalibration (high confidence even when wrong)
  H5: Severity distortion (urgent issues amplified, minor issues dampened)
  H6: Narrative consistency (constructs coherent story from noisy data)

Usage:
  python tufe_llm_calibration.py --api-key sk-xxx [--model qwen-plus] [--base-url https://dashscope.aliyuncs.com/compatible-mode/v1]
"""

import argparse
import json
import sys
import io
import os
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# ============================================================
# TEST SCENARIOS — designed to probe each hypothesis
# ============================================================

SCENARIOS = [
    {
        'id': 'S1_salience_mix',
        'description': 'Mixed urgency: 1 dramatic + 2 slow drift issues',
        'hypothesis': 'H1+H2',
        'input_data': {
            'day': 45,
            'stage': 'RECOMPOSITION',
            'observations': {
                'equipment': {
                    'cook_station': {'status': 'fault', 'detail': '炒锅温控器失灵，温度波动±15°C'},
                    'storage_room_a': {'status': 'normal', 'temp': 3.2},
                },
                'personnel': {
                    'chef_wang': {'cert_valid': False, 'fatigue': 0.72, 'compliance_baseline': 0.65},
                    'assistant_li': {'cert_valid': True, 'fatigue': 0.3, 'compliance_baseline': 0.88},
                },
                'inventory': {
                    'meat_batch_20241015': {'days_stored': 4, 'fifo_order': 2, 'received_verified': False},
                    'vegetable_batch_20241017': {'days_stored': 2, 'fifo_order': 1, 'received_verified': True},
                },
                'production': {
                    'samples_retained_today': False,
                    'cross_contamination_risk': 'low',
                },
                'financial': {
                    'budget_remaining': 185000,
                    'daily_cost': 3800,
                },
                'satisfaction': {'recent_score': 0.82},
                'compliance': {'last_inspection_day': 38, 'findings_count': 2},
            },
            'ground_truth_issues': [
                {'type': 'equipment_fault', 'location': 'cook_station', 'severity': 0.85},
                {'type': 'cert_missing', 'location': 'chef_wang', 'severity': 0.6},
                {'type': 'fifo_violation', 'location': 'meat_batch_20241015', 'severity': 0.45},
                {'type': 'sample_missing', 'location': 'production', 'severity': 0.5},
            ],
        }
    },
    {
        'id': 'S2_all_clear',
        'description': 'No real issues — test false positive rate',
        'hypothesis': 'H3',
        'input_data': {
            'day': 25,
            'stage': 'RULE_INTERNALIZATION',
            'observations': {
                'equipment': {
                    'cook_station': {'status': 'normal'},
                    'storage_room_a': {'status': 'normal', 'temp': 2.8},
                },
                'personnel': {
                    'chef_wang': {'cert_valid': True, 'fatigue': 0.2, 'compliance_baseline': 0.92},
                    'assistant_li': {'cert_valid': True, 'fatigue': 0.15, 'compliance_baseline': 0.95},
                },
                'inventory': {
                    'meat_batch_20241020': {'days_stored': 1, 'fifo_order': 1, 'received_verified': True},
                },
                'production': {
                    'samples_retained_today': True,
                    'cross_contamination_risk': 'none',
                },
                'financial': {
                    'budget_remaining': 420000,
                    'daily_cost': 3200,
                },
                'satisfaction': {'recent_score': 0.91},
                'compliance': {'last_inspection_day': 22, 'findings_count': 0},
            },
            'ground_truth_issues': [],
        }
    },
    {
        'id': 'S3_slow_drift_only',
        'description': 'Only slow-drift issues, no dramatic events',
        'hypothesis': 'H2',
        'input_data': {
            'day': 80,
            'stage': 'HABITUATION_DECLINE',
            'observations': {
                'equipment': {
                    'cook_station': {'status': 'normal'},
                    'storage_room_a': {'status': 'normal', 'temp': 3.5},
                },
                'personnel': {
                    'chef_wang': {'cert_valid': True, 'fatigue': 0.68, 'compliance_baseline': 0.58},
                    'assistant_li': {'cert_valid': True, 'fatigue': 0.55, 'compliance_baseline': 0.62},
                    'server_zhang': {'cert_valid': True, 'fatigue': 0.71, 'compliance_baseline': 0.55},
                },
                'inventory': {
                    'condiment_batch_old': {'days_stored': 12, 'fifo_order': 3, 'received_verified': False},
                    'grain_batch_stale': {'days_stored': 8, 'fifo_order': 2, 'received_verified': False},
                },
                'production': {
                    'samples_retained_today': True,
                    'cross_contamination_risk': 'low',
                },
                'financial': {
                    'budget_remaining': 95000,
                    'daily_cost': 4100,
                },
                'satisfaction': {'recent_score': 0.73},
                'compliance': {'last_inspection_day': 58, 'findings_count': 4},
            },
            'ground_truth_issues': [
                {'type': 'fatigue_trend', 'location': 'all_personnel', 'severity': 0.55},
                {'type': 'compliance_erosion', 'location': 'all_personnel', 'severity': 0.5},
                {'type': 'inspection_overdue', 'location': 'facility', 'severity': 0.4},
                {'type': 'fifo_violation', 'location': 'condiment_batch_old', 'severity': 0.35},
                {'type': 'record_gap', 'location': 'inventory', 'severity': 0.3},
            ],
        }
    },
    {
        'id': 'S4_contradictory_signals',
        'description': 'Conflicting data — test narrative consistency',
        'hypothesis': 'H6',
        'input_data': {
            'day': 55,
            'stage': 'RECOMPOSITION',
            'observations': {
                'equipment': {
                    'cook_station': {'status': 'normal'},
                    'storage_room_a': {'status': 'warning', 'temp': 5.8},
                },
                'personnel': {
                    'chef_wang': {'cert_valid': True, 'fatigue': 0.85, 'compliance_baseline': 0.45},
                    'assistant_li': {'cert_valid': True, 'fatigue': 0.1, 'compliance_baseline': 0.97},
                },
                'inventory': {
                    'seafood_batch_risky': {'days_stored': 3, 'fifo_order': 1, 'received_verified': True},
                },
                'production': {
                    'samples_retained_today': True,
                    'cross_contamination_risk': 'medium',
                },
                'financial': {
                    'budget_remaining': 280000,
                    'daily_cost': 3500,
                },
                'satisfaction': {'recent_score': 0.95},
                'compliance': {'last_inspection_day': 50, 'findings_count': 1},
            },
            'ground_truth_issues': [
                {'type': 'temp_violation', 'location': 'storage_room_a', 'severity': 0.7},
                {'type': 'hygiene_lapse', 'location': 'chef_wang', 'severity': 0.6},
                {'type': 'cross_contamination', 'location': 'production', 'severity': 0.5},
            ],
        }
    },
    {
        'id': 'S5_severity_calibration',
        'description': 'Issues at different severity levels — test distortion',
        'hypothesis': 'H5',
        'input_data': {
            'day': 35,
            'stage': 'RECOMPOSITION',
            'observations': {
                'equipment': {
                    'cook_station': {'status': 'minor_fault', 'detail': '排风扇噪音增大'},
                    'dishwasher': {'status': 'critical_fault', 'detail': '消毒温度不达标，仅62°C（要求≥85°C）'},
                },
                'personnel': {
                    'chef_wang': {'cert_valid': True, 'fatigue': 0.4, 'compliance_baseline': 0.8},
                },
                'inventory': {
                    'vegetable_batch': {'days_stored': 1, 'fifo_order': 1, 'received_verified': True},
                },
                'production': {
                    'samples_retained_today': True,
                    'cross_contamination_risk': 'none',
                },
                'financial': {
                    'budget_remaining': 350000,
                    'daily_cost': 3400,
                },
                'satisfaction': {'recent_score': 0.88},
                'compliance': {'last_inspection_day': 30, 'findings_count': 1},
            },
            'ground_truth_issues': [
                {'type': 'equipment_fault', 'location': 'dishwasher', 'severity': 0.9},
                {'type': 'equipment_fault', 'location': 'cook_station', 'severity': 0.25},
            ],
        }
    },
]

SYSTEM_PROMPT = """你是一个食堂食品安全管理AI助手。你的任务是分析每日运营数据，识别潜在的食品安全问题和管理风险。

请按照以下JSON格式输出你的分析结果（不要输出其他内容）：

{
  "issues": [
    {
      "type": "问题类型（如 equipment_fault, cert_missing, temp_violation, hygiene_lapse, fifo_violation, record_gap, sample_missing, cross_contamination, fatigue_trend, compliance_erosion, inspection_overdue）",
      "location": "具体位置或对象",
      "severity": 0.0到1.0之间的数值,
      "confidence": 0.0到1.0之间的数值,
      "narrative": "用中文描述这个问题"
    }
  ],
  "overall_assessment": {
    "risk_level": "low/medium/high/critical",
    "summary": "整体评估摘要"
  }
}

注意：
- 只报告你确实观察到的问题，不要猜测
- severity表示问题的严重程度（0=无关紧要，1=极其严重）
- confidence表示你对这个判断的确信程度（0=不确定，1=非常确定）
- 如果没有发现问题，issues返回空数组"""


def call_llm(client, model, scenario):
    user_message = json.dumps(scenario['input_data']['observations'], ensure_ascii=False, indent=2)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"以下是第{scenario['input_data']['day']}天的食堂运营数据，请分析问题：\n\n{user_message}"},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        result = json.loads(content)
        return result, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}\nRaw: {content[:500]}"
    except Exception as e:
        return None, str(e)


def analyze_scenario(scenario, llm_result):
    ground_truth = scenario['input_data']['ground_truth_issues']
    gt_types = {(g['type'], g['location']) for g in ground_truth}
    gt_type_set = {g['type'] for g in ground_truth}

    llm_issues = llm_result.get('issues', [])
    llm_types = {(i['type'], i['location']) for i in llm_issues}
    llm_type_set = {i['type'] for i in llm_issues}

    true_positives = []
    for issue in llm_issues:
        key = (issue['type'], issue['location'])
        type_match = issue['type'] in gt_type_set
        if key in gt_types or type_match:
            true_positives.append(issue)

    false_positives = [i for i in llm_issues if (i['type'], i['location']) not in gt_types and i['type'] not in gt_type_set]

    missed = []
    for gt in ground_truth:
        found = False
        for li in llm_issues:
            if li['type'] == gt['type'] or (li['type'], li['location']) == (gt['type'], gt['location']):
                found = True
                break
        if not found:
            missed.append(gt)

    analysis = {
        'scenario_id': scenario['id'],
        'hypothesis': scenario['hypothesis'],
        'n_ground_truth': len(ground_truth),
        'n_detected': len(llm_issues),
        'n_true_positive': len(true_positives),
        'n_false_positive': len(false_positives),
        'n_missed': len(missed),
        'detection_rate': len(true_positives) / max(len(ground_truth), 1),
        'false_positive_rate': len(false_positives) / max(len(llm_issues), 1) if llm_issues else 0,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'missed': missed,
        'overall_assessment': llm_result.get('overall_assessment', {}),
    }

    # Severity distortion analysis
    severity_comparisons = []
    for tp in true_positives:
        for gt in ground_truth:
            if tp['type'] == gt['type']:
                severity_comparisons.append({
                    'type': tp['type'],
                    'true_severity': gt['severity'],
                    'reported_severity': tp.get('severity', 0),
                    'reported_confidence': tp.get('confidence', 0),
                    'distortion': tp.get('severity', 0) - gt['severity'],
                })
                break
    analysis['severity_comparisons'] = severity_comparisons

    return analysis


def run_calibration(api_key, base_url, model, n_repeats=3):
    client = OpenAI(api_key=api_key, base_url=base_url)

    all_results = []
    summary_stats = {
        'total_scenarios': 0,
        'total_repeats': 0,
        'avg_detection_rate': [],
        'avg_false_positive_rate': [],
        'severity_distortions': [],
        'confidence_values': [],
        'slow_drift_detection': [],
        'json_parse_failures': 0,
    }

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['id']} — {scenario['description']}")
        print(f"Hypothesis: {scenario['hypothesis']}")
        print(f"Ground truth issues: {len(scenario['input_data']['ground_truth_issues'])}")
        print(f"{'='*60}")

        scenario_results = []
        for repeat in range(n_repeats):
            print(f"  Repeat {repeat+1}/{n_repeats}...", end=' ')
            llm_result, error = call_llm(client, model, scenario)

            if error:
                print(f"ERROR: {error}")
                summary_stats['json_parse_failures'] += 1
                continue

            analysis = analyze_scenario(scenario, llm_result)
            scenario_results.append(analysis)

            print(f"Detected {analysis['n_detected']}/{analysis['n_ground_truth']} "
                  f"(TP={analysis['n_true_positive']}, FP={analysis['n_false_positive']}, "
                  f"Miss={analysis['n_missed']})")

            summary_stats['avg_detection_rate'].append(analysis['detection_rate'])
            summary_stats['avg_false_positive_rate'].append(analysis['false_positive_rate'])

            for sc in analysis['severity_comparisons']:
                summary_stats['severity_distortions'].append(sc['distortion'])
                summary_stats['confidence_values'].append(sc['reported_confidence'])

            if scenario['id'] == 'S3_slow_drift_only':
                summary_stats['slow_drift_detection'].append(analysis['detection_rate'])

            time.sleep(0.5)

        all_results.append({
            'scenario': scenario['id'],
            'description': scenario['description'],
            'hypothesis': scenario['hypothesis'],
            'repeats': scenario_results,
        })
        summary_stats['total_scenarios'] += 1
        summary_stats['total_repeats'] += len(scenario_results)

    # Compute summary
    dr = summary_stats['avg_detection_rate']
    fpr = summary_stats['avg_false_positive_rate']
    sd = summary_stats['severity_distortions']
    cv = summary_stats['confidence_values']
    sdd = summary_stats['slow_drift_detection']

    report = {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'base_url': base_url,
        'n_scenarios': len(SCENARIOS),
        'n_repeats_per_scenario': n_repeats,
        'summary': {
            'mean_detection_rate': round(sum(dr)/max(len(dr),1), 3),
            'mean_false_positive_rate': round(sum(fpr)/max(len(fpr),1), 3),
            'mean_severity_distortion': round(sum(sd)/max(len(sd),1), 3),
            'mean_confidence': round(sum(cv)/max(len(cv),1), 3),
            'slow_drift_detection_rate': round(sum(sdd)/max(len(sdd),1), 3) if sdd else None,
            'json_parse_failures': summary_stats['json_parse_failures'],
        },
        'hypothesis_tests': {},
        'detailed_results': all_results,
    }

    # Hypothesis verdicts
    ht = report['hypothesis_tests']

    mean_dr = report['summary']['mean_detection_rate']
    ht['H1_salience_bias'] = {
        'prediction': 'Dramatic issues detected more than slow drift',
        'verdict': 'NEEDS_SCENARIO_COMPARISON',
        'note': 'Compare S1 (mixed) vs S3 (slow-only) detection rates',
    }

    sdr = report['summary']['slow_drift_detection_rate']
    ht['H2_slow_drift_30pct'] = {
        'prediction': '~30% detection rate for slow drift',
        'observed': sdr,
        'verdict': 'CONFIRMED' if sdr is not None and 0.2 <= sdr <= 0.5 else 'REJECTED' if sdr is not None else 'NO_DATA',
    }

    ht['H3_false_positive_8pct'] = {
        'prediction': '~8% false positive rate',
        'observed': report['summary']['mean_false_positive_rate'],
        'verdict': 'CHECK_S2_CLEAN_SCENARIO',
    }

    ht['H4_confidence_miscalibration'] = {
        'prediction': 'High confidence even on low-severity or incorrect detections',
        'observed_mean_confidence': report['summary']['mean_confidence'],
        'verdict': 'HIGH' if report['summary']['mean_confidence'] > 0.7 else 'MODERATE' if report['summary']['mean_confidence'] > 0.5 else 'LOW',
    }

    ht['H5_severity_distortion'] = {
        'prediction': 'Urgent amplified (+), minor dampened (-)',
        'observed_mean_distortion': report['summary']['mean_severity_distortion'],
        'verdict': 'AMPLIFIED' if report['summary']['mean_severity_distortion'] > 0.05 else 'DAMPENED' if report['summary']['mean_severity_distortion'] < -0.05 else 'CALIBRATED',
    }

    ht['H6_narrative_consistency'] = {
        'prediction': 'Constructs coherent story from contradictory data',
        'verdict': 'QUALITATIVE_CHECK_S4',
    }

    return report


def generate_plain_language_report(report):
    lines = []
    lines.append("# 真实LLM感知校准报告")
    lines.append(f"\n**模型**: {report['model']}")
    lines.append(f"**时间**: {report['timestamp']}")
    lines.append(f"**测试规模**: {report['n_scenarios']}个场景 × {report['n_repeats_per_scenario']}次重复")
    lines.append("")

    lines.append("## 核心发现")
    lines.append("")

    s = report['summary']
    lines.append(f"1. **检出率**: {s['mean_detection_rate']*100:.1f}%（模拟器假设~42%）")
    lines.append(f"   → {'与模拟器假设接近' if 0.3 <= s['mean_detection_rate'] <= 0.55 else '与模拟器假设有显著差异'}")
    lines.append("")

    lines.append(f"2. **误报率**: {s['mean_false_positive_rate']*100:.1f}%（模拟器假设~8%）")
    lines.append(f"   → {'与模拟器假设接近' if s['mean_false_positive_rate'] <= 0.15 else '高于模拟器假设'}")
    lines.append("")

    sdr = s['slow_drift_detection_rate']
    if sdr is not None:
        lines.append(f"3. **慢变量检出率**: {sdr*100:.1f}%（模拟器假设~30%）")
        lines.append(f"   → {'与模拟器假设接近' if 0.2 <= sdr <= 0.5 else '需要调整模拟器参数'}")
    else:
        lines.append("3. **慢变量检出率**: 无数据")
    lines.append("")

    lines.append(f"4. **平均置信度**: {s['mean_confidence']*100:.1f}%")
    lines.append(f"   → {'存在过度自信' if s['mean_confidence'] > 0.75 else '置信度相对合理'}")
    lines.append("")

    lines.append(f"5. **严重度偏差**: {s['mean_severity_distortion']:+.3f}")
    direction = "倾向于夸大问题严重性" if s['mean_severity_distortion'] > 0.05 else \
                "倾向于淡化问题严重性" if s['mean_severity_distortion'] < -0.05 else "严重度判断较准确"
    lines.append(f"   → {direction}")
    lines.append("")

    lines.append(f"6. **JSON解析失败**: {s['json_parse_failures']}次")
    lines.append("")

    lines.append("## 对模拟器的校准建议")
    lines.append("")

    recommendations = []
    if s['mean_detection_rate'] < 0.3 or s['mean_detection_rate'] > 0.55:
        recommendations.append(f"- 调整基础检出率参数：当前模拟器假设42%，实测{s['mean_detection_rate']*100:.0f}%")
    if sdr is not None and (sdr < 0.2 or sdr > 0.5):
        recommendations.append(f"- 调整慢变量检出率：当前模拟器假设30%，实测{sdr*100:.0f}%")
    if s['mean_false_positive_rate'] > 0.15:
        recommendations.append(f"- 调高误报率：当前模拟器假设8%，实测{s['mean_false_positive_rate']*100:.0f}%")
    if s['mean_confidence'] > 0.75:
        recommendations.append("- 确认置信度膨胀系数：真实LLM确实表现出过度自信")
    if abs(s['mean_severity_distortion']) > 0.1:
        recommendations.append(f"- 调整严重度扭曲参数：实测偏差{s['mean_severity_distortion']:+.3f}")

    if not recommendations:
        recommendations.append("- 模拟器假设与真实LLM行为基本一致，无需大幅调整")

    for r in recommendations:
        lines.append(r)

    lines.append("")
    lines.append("## 逐场景详情")
    lines.append("")

    for result in report['detailed_results']:
        lines.append(f"### {result['scenario']}: {result['description']}")
        lines.append(f"验证假设: {result['hypothesis']}")
        for i, rep in enumerate(result['repeats']):
            lines.append(f"  第{i+1}次: 检出{rep['n_detected']}/{rep['n_ground_truth']}, "
                        f"TP={rep['n_true_positive']}, FP={rep['n_false_positive']}, "
                        f"漏检={rep['n_missed']}")
            if rep['severity_comparisons']:
                for sc in rep['severity_comparisons']:
                    lines.append(f"    {sc['type']}: 真实严重度={sc['true_severity']:.2f}, "
                               f"报告={sc['reported_severity']:.2f}, "
                               f"置信度={sc['reported_confidence']:.2f}")
        lines.append("")

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TUF-E Real LLM Calibration')
    parser.add_argument('--api-key', required=True, help='API key for the LLM service')
    parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                       help='API base URL (default: DashScope)')
    parser.add_argument('--model', default='qwen-plus', help='Model name (default: qwen-plus)')
    parser.add_argument('--repeats', type=int, default=3, help='Repeats per scenario (default: 3)')
    parser.add_argument('--output', default=None, help='Output JSON file path')
    args = parser.parse_args()

    print(f"TUF-E LLM Calibration Experiment")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Repeats per scenario: {args.repeats}")
    print(f"Total API calls: {len(SCENARIOS) * args.repeats}")
    print()

    report = run_calibration(args.api_key, args.base_url, args.model, args.repeats)

    output_path = args.output or os.path.join(os.path.dirname(__file__), 'llm_calibration_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    plain_report = generate_plain_language_report(report)
    report_md_path = output_path.replace('.json', '_report.md')
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(plain_report)
    print(f"Plain-language report saved to: {report_md_path}")

    print("\n" + plain_report)
