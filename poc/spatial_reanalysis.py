"""
空间视角重分析: 从神经元作用机理审视TUF-E接地实验

核心问题: 智能体的学习过程在"操作空间"中是否展现出类似
神经系统的空间组织特征？

神经元的关键空间特性:
1. 拓扑映射(Topographic mapping): 功能相似的神经元在物理空间中邻近
2. 感受野(Receptive field): 每个神经元只对输入空间的局部区域响应
3. 侧抑制(Lateral inhibition): 活跃神经元抑制邻居，增强对比度
4. Hebbian集群: "一起放电的神经元连在一起"形成cell assembly
5. 空间巩固: 记忆不是存在单个神经元中，而是分布在空间模式中

将操作视为"神经元"，操作偏好向量视为"激活模式"，
分析学习过程中的空间动力学。
"""

import json
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OP_NAMES = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div', 4: 'neg', 5: 'abs', 6: 'sqr', 7: 'id'}


def load_results():
    with open('grounding_experiment_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_spatial_structure(results):
    print("=" * 70)
    print("空间视角重分析: TUF-E接地实验")
    print("=" * 70)

    episodes = results['episodes']
    milestones = results['milestones']
    n_ops = 8

    # === 1. 操作空间的拓扑映射 ===
    print("\n--- 1. 操作空间拓扑映射 ---")
    print("问题: 功能相似的操作是否在偏好空间中聚集？")

    # 按算术性质分组
    binary_ops = {0, 1, 2, 3}   # add, sub, mul, div (二元运算)
    unary_ops = {4, 5, 6, 7}    # neg, abs, sqr, id (一元运算)

    final_prefs = milestones[-1]['op_preferences'] if milestones else {}
    binary_pref = sum(final_prefs.get(str(k), 0) for k in binary_ops)
    unary_pref = sum(final_prefs.get(str(k), 0) for k in unary_ops)

    print(f"  二元运算(add/sub/mul/div)总偏好: {binary_pref:.4f}")
    print(f"  一元运算(neg/abs/sqr/id)总偏好: {unary_pref:.4f}")
    print(f"  分离比: {binary_pref / max(unary_pref, 0.001):.2f}x")

    # 检查学习过程中二元vs一元的分化轨迹
    print("\n  分化轨迹:")
    for m in milestones:
        bp = sum(m['op_preferences'].get(str(k), 0) for k in binary_ops)
        up = sum(m['op_preferences'].get(str(k), 0) for k in unary_ops)
        ep = m['total_attempts']
        print(f"    Ep {ep:>3}: binary={bp:.3f}, unary={up:.3f}, ratio={bp/max(up,0.001):.1f}x")

    # === 2. 感受野分析 ===
    print("\n--- 2. 感受野分析 ---")
    print("问题: 每个操作是否对特定的输入模式有选择性响应？")

    # 统计每个操作在不同数值范围内的成功率
    op_success_by_range = {i: {'small': [0, 0], 'medium': [0, 0], 'large': [0, 0]} for i in range(n_ops)}

    for ep in episodes:
        chosen = ep['chosen_ops']
        target = abs(ep['target'])
        correct = ep['correct']

        if target < 10:
            r = 'small'
        elif target < 50:
            r = 'medium'
        else:
            r = 'large'

        for op in chosen:
            if op in op_success_by_range:
                op_success_by_range[op][r][1] += 1
                if correct:
                    op_success_by_range[op][r][0] += 1

    print(f"  {'操作':>8} | {'小(<10)':>10} | {'中(10-50)':>10} | {'大(>50)':>10}")
    print(f"  " + "-" * 48)
    for op in range(n_ops):
        rates = []
        for r in ['small', 'medium', 'large']:
            s, t = op_success_by_range[op][r]
            rate = f"{s}/{t}={s/max(t,1)*100:.0f}%" if t > 0 else "N/A"
            rates.append(rate)
        print(f"  op_{op}({OP_NAMES[op]:>3}) | {rates[0]:>10} | {rates[1]:>10} | {rates[2]:>10}")

    # === 3. 侧抑制效应 ===
    print("\n--- 3. 侧抑制效应 ---")
    print("问题: 当一个操作的偏好上升时，其他操作是否被系统性抑制？")

    if len(milestones) >= 2:
        pref_changes = []
        for i in range(1, len(milestones)):
            prev = milestones[i-1]['op_preferences']
            curr = milestones[i]['op_preferences']
            changes = {}
            for k in range(n_ops):
                changes[k] = curr.get(str(k), 0) - prev.get(str(k), 0)
            pref_changes.append(changes)

        # 检查: mul上升时，其他操作是否下降
        mul_rises = [pc for pc in pref_changes if pc[2] > 0.01]
        if mul_rises:
            avg_others_when_mul_rises = {}
            for k in range(n_ops):
                if k != 2:
                    vals = [pc[k] for pc in mul_rises]
                    avg_others_when_mul_rises[k] = np.mean(vals)

            print(f"  当mul偏好上升时({len(mul_rises)}次):")
            for k, v in sorted(avg_others_when_mul_rises.items(), key=lambda x: x[1]):
                direction = "↓" if v < -0.005 else ("↑" if v > 0.005 else "→")
                print(f"    op_{k}({OP_NAMES[k]:>3}): {v:+.4f} {direction}")

            inhibited = sum(1 for v in avg_others_when_mul_rises.values() if v < -0.005)
            print(f"  被抑制的操作数: {inhibited}/{n_ops-1}")
            print(f"  侧抑制判定: {'存在' if inhibited >= 3 else '弱/不存在'}")

    # === 4. Hebbian集群检测 ===
    print("\n--- 4. Hebbian集群检测 ---")
    print("问题: 是否存在'一起成功'的操作组？")

    # 共现矩阵: 哪些操作在同一episode中被选择且成功
    cooccurrence = np.zeros((n_ops, n_ops))
    for ep in episodes:
        if ep['correct']:
            ops_used = ep['chosen_ops']
            for i in range(len(ops_used)):
                for j in range(i+1, len(ops_used)):
                    a, b = ops_used[i], ops_used[j]
                    if a < n_ops and b < n_ops:
                        cooccurrence[a][b] += 1
                        cooccurrence[b][a] += 1

    print("  成功共现矩阵:")
    header = "         " + " ".join(f"op_{i}" for i in range(n_ops))
    print(header)
    for i in range(n_ops):
        row = f"  op_{i}({OP_NAMES[i]:>3}) " + " ".join(f"{cooccurrence[i][j]:>4.0f}" for j in range(n_ops))
        print(row)

    # 找最强的共现对
    pairs = []
    for i in range(n_ops):
        for j in range(i+1, n_ops):
            if cooccurrence[i][j] > 0:
                pairs.append((i, j, cooccurrence[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  最强Hebbian关联:")
    for a, b, count in pairs[:5]:
        print(f"    op_{a}({OP_NAMES[a]}) ↔ op_{b}({OP_NAMES[b]}): {count}次共同成功")

    # === 5. 空间巩固分析 ===
    print("\n--- 5. 空间巩固分析 ---")
    print("问题: 知识是集中在少数操作上，还是分布式的？")

    if milestones:
        final = milestones[-1]['op_preferences']
        prefs_array = np.array([final.get(str(k), 0) for k in range(n_ops)])

        # Shannon entropy of preference distribution
        nonzero = prefs_array[prefs_array > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) > 0 else 0
        max_entropy = np.log2(n_ops)
        concentration = 1.0 - entropy / max_entropy

        print(f"  偏好分布熵: {entropy:.3f} bits (最大={max_entropy:.3f})")
        print(f"  集中度: {concentration:.3f} (0=均匀, 1=完全集中)")
        print(f"  有效操作数: {2**entropy:.1f} (等价于多少个均匀分布的操作)")

        # Gini coefficient
        sorted_prefs = np.sort(prefs_array)
        n = len(sorted_prefs)
        gini = sum((2*i - n - 1) * sorted_prefs[i] for i in range(n)) / (n * sum(sorted_prefs)) if sum(sorted_prefs) > 0 else 0
        print(f"  Gini系数: {gini:.3f} (0=平等, 1=完全不平等)")

        # 与早期对比
        early = milestones[0]['op_preferences']
        early_array = np.array([early.get(str(k), 0) for k in range(n_ops)])
        early_nonzero = early_array[early_array > 0]
        early_entropy = -np.sum(early_nonzero * np.log2(early_nonzero)) if len(early_nonzero) > 0 else 0
        early_concentration = 1.0 - early_entropy / max_entropy

        print(f"\n  巩固轨迹:")
        print(f"    初始: 熵={early_entropy:.3f}, 集中度={early_concentration:.3f}")
        print(f"    最终: 熵={entropy:.3f}, 集中度={concentration:.3f}")
        print(f"    变化: Δ熵={entropy-early_entropy:+.3f}, Δ集中度={concentration-early_concentration:+.3f}")

    # === 6. 与神经科学的结构性比对 ===
    print(f"\n{'=' * 70}")
    print("结构性比对总结")
    print(f"{'=' * 70}")

    findings = []

    # Check topographic separation
    if binary_pref / max(unary_pref, 0.001) > 2.0:
        findings.append(("拓扑映射", "CONFIRMED", f"二元/一元分离比={binary_pref/max(unary_pref,0.001):.1f}x"))
    else:
        findings.append(("拓扑映射", "WEAK", f"二元/一元分离比={binary_pref/max(unary_pref,0.001):.1f}x"))

    # Check lateral inhibition
    if mul_rises and sum(1 for v in avg_others_when_mul_rises.values() if v < -0.005) >= 3:
        findings.append(("侧抑制", "CONFIRMED", f"mul上升时{sum(1 for v in avg_others_when_mul_rises.values() if v < -0.005)}个操作被抑制"))
    else:
        findings.append(("侧抑制", "WEAK", "未检测到系统性抑制"))

    # Check Hebbian assemblies
    if pairs and pairs[0][2] >= 3:
        findings.append(("Hebbian集群", "CONFIRMED", f"最强共现对: op_{pairs[0][0]}↔op_{pairs[0][1]} ({pairs[0][2]}次)"))
    else:
        findings.append(("Hebbian集群", "ABSENT", "无显著共现模式"))

    # Check concentration increase
    if milestones and concentration > early_concentration + 0.1:
        findings.append(("空间巩固", "CONFIRMED", f"集中度从{early_concentration:.3f}增至{concentration:.3f}"))
    else:
        findings.append(("空间巩固", "WEAK", "集中度变化不明显"))

    for name, status, detail in findings:
        marker = "✓" if status == "CONFIRMED" else ("?" if status == "WEAK" else "✗")
        print(f"  [{marker}] {name}: {status} — {detail}")

    confirmed = sum(1 for _, s, _ in findings if s == "CONFIRMED")
    print(f"\n  确认的神经样特征: {confirmed}/{len(findings)}")

    return findings


if __name__ == '__main__':
    results = load_results()
    analyze_spatial_structure(results)
