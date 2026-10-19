"""
v1.2 顿悟(相变)深度分析脚本

回答三个问题:
1. 顿悟的时间分布规律是什么?
2. 顿悟前后行为是否有可测量变化? (真顿悟的行为证据)
3. 随机对照下"假顿悟"的频率是多少? (基线排除)

[ME-C4] 本脚本为分析工具，不是实验主体。所有结论标注置信度。
"""

import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
from tufe_grounding_v12 import (
    SharedEnvironment, MultiAgentSystem, ConsensusKnowledgeStore
)


def run_with_detailed_tracking(seed, n_episodes=5000):
    """运行条件A并收集每轮的详细数据"""
    env = SharedEnvironment(seed=seed, feedback_delay=3)
    system = MultiAgentSystem(use_consensus=True, use_pivots=True, seed=seed)
    pending = {}

    episode_data = []

    for ep in range(n_episodes):
        r = np.random.default_rng(ep + seed).random()
        difficulty = 1 if r < 0.5 else (2 if r < 0.8 else 3)
        task = env.generate_task(difficulty)
        pending[ep] = task

        fb_batch, agent_idx = system.run_episode(env, task, ep, pending)
        correct_this_ep = False
        if fb_batch:
            learn_result = system.process_feedback(fb_batch, pending, ep)
            for fb in fb_batch:
                if fb.get('correct', False):
                    correct_this_ep = True
                pending.pop(fb['episode'], None)

            if learn_result:
                pt = learn_result.get('phase_transition')
                confirmed = system.consensus_store.get_confirmed_pivots()
                volume = learn_result.get('feasible_volume', 0)

                episode_data.append({
                    'episode': ep,
                    'correct': correct_this_ep,
                    'feasible_volume': volume,
                    'n_confirmed': len(confirmed),
                    'has_phase_transition': pt is not None,
                    'transition_detail': pt,
                    'criticality': system.phase_detector.criticality_history[-1]['criticality'] if system.phase_detector.criticality_history else 0,
                })

    remaining = env.flush_feedback()
    if remaining:
        system.process_feedback(remaining, pending, n_episodes)

    return episode_data


def analyze_temporal_distribution(all_episode_data, n_episodes):
    """分析1: 顿悟的时间分布"""
    print("=" * 60)
    print("分析1: 顿悟的时间分布")
    print("=" * 60)

    all_transitions = []
    for seed_data in all_episode_data:
        for ed in seed_data:
            if ed['has_phase_transition']:
                all_transitions.append(ed['episode'])

    if not all_transitions:
        print("  未检测到任何顿悟事件")
        return

    print(f"  总顿悟次数: {len(all_transitions)}")
    print(f"  最早发生: 第 {min(all_transitions)} 轮")
    print(f"  最晚发生: 第 {max(all_transitions)} 轮")
    print(f"  中位数: 第 {int(np.median(all_transitions))} 轮")

    # 分四段看分布
    quarter = n_episodes // 4
    bins = [0, quarter, quarter*2, quarter*3, n_episodes]
    bin_labels = ['前25%', '25-50%', '50-75%', '后25%']
    counts = [0, 0, 0, 0]
    for t in all_transitions:
        for i in range(4):
            if bins[i] <= t < bins[i+1]:
                counts[i] += 1
                break

    print(f"\n  分段分布:")
    for label, count in zip(bin_labels, counts):
        bar = "█" * (count // 2)
        print(f"    {label}: {count:3d} 次 {bar}")

    # 相邻顿悟间隔
    sorted_t = sorted(all_transitions)
    intervals = [sorted_t[i+1] - sorted_t[i] for i in range(len(sorted_t)-1)]
    if intervals:
        print(f"\n  相邻顿悟间隔:")
        print(f"    平均: {np.mean(intervals):.0f} 轮")
        print(f"    中位数: {np.median(intervals):.0f} 轮")
        print(f"    最短: {min(intervals)} 轮")
        print(f"    最长: {max(intervals)} 轮")

    # Betti变化类型统计
    b0_changes = defaultdict(int)
    b1_changes = defaultdict(int)
    for seed_data in all_episode_data:
        for ed in seed_data:
            if ed['has_phase_transition'] and ed['transition_detail']:
                td = ed['transition_detail']
                b0_changes[td['betti_0_change']] += 1
                b1_changes[td['betti_1_change']] += 1

    print(f"\n  Betti-0 (连通分量) 变化分布:")
    for delta, count in sorted(b0_changes.items()):
        direction = "合并" if delta < 0 else ("碎片化" if delta > 0 else "不变")
        print(f"    Δ={delta:+d} ({direction}): {count} 次")

    print(f"\n  Betti-1 (环数) 变化分布:")
    for delta, count in sorted(b1_changes.items()):
        direction = "新环形成" if delta > 0 else ("环解散" if delta < 0 else "不变")
        print(f"    Δ={delta:+d} ({direction}): {count} 次")


def analyze_behavioral_impact(all_episode_data, window=50):
    """分析2: 顿悟前后的行为变化"""
    print("\n" + "=" * 60)
    print(f"分析2: 顿悟前后的行为变化 (窗口={window}轮)")
    print("=" * 60)

    before_sr = []
    after_sr = []
    before_vol_change = []
    after_vol_change = []

    for seed_data in all_episode_data:
        episodes = [ed['episode'] for ed in seed_data]
        corrects = [ed['correct'] for ed in seed_data]
        volumes = [ed['feasible_volume'] for ed in seed_data]
        transitions = [i for i, ed in enumerate(seed_data) if ed['has_phase_transition']]

        for t_idx in transitions:
            ep_val = seed_data[t_idx]['episode']

            # 找窗口内的索引
            before_indices = [i for i, e in enumerate(episodes)
                            if ep_val - window <= e < ep_val]
            after_indices = [i for i, e in enumerate(episodes)
                           if ep_val < e <= ep_val + window]

            if before_indices and after_indices:
                b_sr = np.mean([corrects[i] for i in before_indices])
                a_sr = np.mean([corrects[i] for i in after_indices])
                before_sr.append(b_sr)
                after_sr.append(a_sr)

                b_vol = np.mean([volumes[i] for i in before_indices])
                a_vol = np.mean([volumes[i] for i in after_indices])
                before_vol_change.append(b_vol)
                after_vol_change.append(a_vol)

    if not before_sr:
        print("  没有足够数据计算行为变化")
        return

    sr_diff = [a - b for a, b in zip(after_sr, before_sr)]
    vol_diff = [a - b for a, b in zip(after_vol_change, before_vol_change)]

    print(f"  样本量: {len(sr_diff)} 次顿悟事件")
    print(f"\n  成功率变化 (后{window}轮 - 前{window}轮):")
    print(f"    平均变化: {np.mean(sr_diff)*100:+.2f}%")
    print(f"    中位数变化: {np.median(sr_diff)*100:+.2f}%")
    improved = sum(1 for d in sr_diff if d > 0)
    worsened = sum(1 for d in sr_diff if d < 0)
    unchanged = sum(1 for d in sr_diff if d == 0)
    print(f"    提升: {improved} 次 | 下降: {worsened} 次 | 不变: {unchanged} 次")

    print(f"\n  可行空间体积变化:")
    print(f"    平均变化: {np.mean(vol_diff):+.1f}")
    print(f"    中位数变化: {np.median(vol_diff):+.1f}")
    vol_improved = sum(1 for d in vol_diff if d > 0)
    vol_worsened = sum(1 for d in vol_diff if d < 0)
    print(f"    扩大: {vol_improved} 次 | 缩小: {vol_worsened} 次")

    # 判断标准: 如果顿悟是真的，之后应有可测量的行为改善
    # 至少成功率或可行空间之一应显著提升
    sig_sr = np.mean(sr_diff) > 0.01  # >1%的提升
    sig_vol = np.mean(vol_diff) > 1.0  # >1的体积增长
    print(f"\n  [ME-C4] 判定 (AI推断, 置信度: 中等):")
    print(f"    成功率有显著提升 (>1%): {'是' if sig_sr else '否'}")
    print(f"    可行空间有显著扩大 (>1): {'是' if sig_vol else '否'}")
    if sig_sr or sig_vol:
        print(f"    → 顿悟有行为证据支持，倾向于'真顿悟'")
    else:
        print(f"    → 顿悟缺乏行为证据，可能是拓扑噪声")


def analyze_shuffled_baseline(all_episode_data, n_shuffles=20):
    """分析3: 置换检验 — 保留真实枢机数据，打乱Betti数的时间序列

    原理: 如果顿悟是真的（来自知识结构的真实拓扑变化），那么打乱
    Betti数的时间顺序后，"顿悟"与行为改善的关联应该消失。
    如果打乱后关联仍在，说明检测器捕获的是某种统计伪影。

    [ME-C4] 这是比纯随机图更合理的对照，因为它保留了系统的全部
            学习动力学，仅破坏拓扑信号与行为的时序对应关系。
    """
    print("\n" + "=" * 60)
    print(f"分析3: 置换检验 ({n_shuffles}次打乱)")
    print("=" * 60)

    rng = np.random.default_rng(777)

    # 收集真实的 (顿悟episode, 后续成功率变化) 配对
    real_pairs = []
    for seed_data in all_episode_data:
        episodes = [ed['episode'] for ed in seed_data]
        corrects = [ed['correct'] for ed in seed_data]
        transitions = [i for i, ed in enumerate(seed_data) if ed['has_phase_transition']]

        for t_idx in transitions:
            ep_val = seed_data[t_idx]['episode']
            after_indices = [i for i, e in enumerate(episodes)
                           if ep_val < e <= ep_val + 50]
            before_indices = [i for i, e in enumerate(episodes)
                            if ep_val - 50 <= e < ep_val]
            if after_indices and before_indices:
                b_sr = np.mean([corrects[i] for i in before_indices])
                a_sr = np.mean([corrects[i] for i in after_indices])
                real_pairs.append((ep_val, a_sr - b_sr))

    if not real_pairs:
        print("  没有足够数据")
        return

    real_mean_diff = np.mean([p[1] for p in real_pairs])
    print(f"  真实顿悟后平均成功率变化: {real_mean_diff*100:+.2f}%")
    print(f"  真实顿悟样本量: {len(real_pairs)}")

    # 置换检验: 把所有episode的正确/错误序列保留，但把"哪些episode被标记为顿悟"随机打乱
    shuffled_diffs = []
    for shuffle_i in range(n_shuffles):
        fake_pairs = []
        for seed_data in all_episode_data:
            episodes = [ed['episode'] for ed in seed_data]
            corrects = [ed['correct'] for ed in seed_data]
            n_transitions = sum(1 for ed in seed_data if ed['has_phase_transition'])

            if n_transitions == 0:
                continue

            # 随机选n_transitions个episode作为"假顿悟点"
            fake_transition_eps = set(rng.choice(episodes, size=min(n_transitions, len(episodes)), replace=False))

            for ep_val in fake_transition_eps:
                after_indices = [i for i, e in enumerate(episodes)
                               if ep_val < e <= ep_val + 50]
                before_indices = [i for i, e in enumerate(episodes)
                                if ep_val - 50 <= e < ep_val]
                if after_indices and before_indices:
                    b_sr = np.mean([corrects[i] for i in before_indices])
                    a_sr = np.mean([corrects[i] for i in after_indices])
                    fake_pairs.append(a_sr - b_sr)

        if fake_pairs:
            shuffled_diffs.append(np.mean(fake_pairs))

    if not shuffled_diffs:
        print("  置换检验无法执行")
        return

    shuf_mean = np.mean(shuffled_diffs)
    shuf_std = np.std(shuffled_diffs)

    print(f"\n  打乱后平均成功率变化: {shuf_mean*100:+.2f}% ± {shuf_std*100:.2f}%")
    print(f"  打乱范围: [{min(shuffled_diffs)*100:+.2f}%, {max(shuffled_diffs)*100:+.2f}%]")

    # p-value: 真实值在打乱分布中的位置
    extreme_count = sum(1 for d in shuffled_diffs if d >= real_mean_diff)
    p_value = extreme_count / len(shuffled_diffs)

    if shuf_std > 0:
        z_score = (real_mean_diff - shuf_mean) / shuf_std
    else:
        z_score = float('inf') if real_mean_diff > shuf_mean else 0

    print(f"\n  Z-score: {z_score:.2f}")
    print(f"  p-value (单侧): {p_value:.3f}")

    print(f"\n  [ME-C4] 判定 (AI推断, 置信度: 中等):")
    if p_value < 0.05:
        print(f"    真实顿悟的行为关联显著强于随机 (p<0.05)")
        print(f"    → 顿悟不是统计伪影，拓扑变化与行为改善有因果关联")
    elif p_value < 0.10:
        print(f"    真实顿悟的行为关联边缘显著 (p<0.10)")
        print(f"    → 有倾向性证据，但需更多数据确认")
    else:
        print(f"    真实顿悟的行为关联不显著 (p={p_value:.3f})")
        print(f"    → 当前证据不足以区分真假顿悟")


def analyze_pivot_content(all_episode_data):
    """分析4: 每次顿悟具体发现了/丢失了什么枢机"""
    print("\n" + "=" * 60)
    print("分析4: 顿悟的枢机内容")
    print("=" * 60)

    all_added = []
    all_removed = []
    sg_at_transitions = []

    for seed_idx, seed_data in enumerate(all_episode_data):
        for ed in seed_data:
            if ed['has_phase_transition'] and ed['transition_detail']:
                td = ed['transition_detail']
                added = td.get('pivots_added', [])
                removed = td.get('pivots_removed', [])
                crit = td.get('criticality_at_transition', 0)

                all_added.extend(added)
                all_removed.extend(removed)
                sg_at_transitions.append(crit)

    print(f"  总顿悟中涉及的枢机变化:")
    print(f"    新增枢机总数: {len(all_added)}")
    print(f"    丢失枢机总数: {len(all_removed)}")

    if all_added:
        strengths = [a['strength'] for a in all_added]
        confirmers = [a['n_confirmers'] for a in all_added]
        print(f"\n  新增枢机的强度分布:")
        print(f"    平均: {np.mean(strengths):.3f}")
        print(f"    范围: [{min(strengths):.3f}, {max(strengths):.3f}]")
        print(f"    平均确认者数: {np.mean(confirmers):.1f}")

    # 按操作类型统计新增枢机
    op_counts = defaultdict(int)
    for a in all_added:
        key = a['key']
        if isinstance(key, tuple) and len(key) >= 3:
            op_id = key[1]
            op_names = {0:'add', 1:'sub', 2:'mul', 3:'div', 4:'neg', 5:'abs', 6:'sqr', 7:'id'}
            op_counts[op_names.get(op_id, f'op{op_id}')] += 1

    if op_counts:
        print(f"\n  新增枢机按运算类型分布:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count // 2)
            print(f"    {op:4s}: {count:3d} {bar}")

    if sg_at_transitions:
        print(f"\n  顿悟时刻的临界度(接近确认门槛的候选数):")
        print(f"    平均: {np.mean(sg_at_transitions):.1f}")
        print(f"    范围: [{min(sg_at_transitions)}, {max(sg_at_transitions)}]")


def analyze_criticality_prediction(all_episode_data):
    """分析5: 临界度是否能预测顿悟时机"""
    print("\n" + "=" * 60)
    print("分析5: 临界度与顿悟时机的关系")
    print("=" * 60)

    crit_at_transitions = []
    crit_at_non_transitions = []

    for seed_data in all_episode_data:
        for ed in seed_data:
            crit = ed.get('criticality', 0)
            if ed['has_phase_transition']:
                crit_at_transitions.append(crit)
            else:
                crit_at_non_transitions.append(crit)

    if not crit_at_transitions:
        print("  没有足够数据")
        return

    print(f"  顿悟时刻的临界度: {np.mean(crit_at_transitions):.1f} ± {np.std(crit_at_transitions):.1f}")
    if crit_at_non_transitions:
        print(f"  非顿悟时刻的临界度: {np.mean(crit_at_non_transitions):.1f} ± {np.std(crit_at_non_transitions):.1f}")

    # 看顿悟前N轮的临界度趋势
    window = 10
    crit_trends_before = []
    for seed_data in all_episode_data:
        crit_series = [(ed['episode'], ed.get('criticality', 0)) for ed in seed_data]
        transition_eps = sorted(set(ed['episode'] for ed in seed_data if ed['has_phase_transition']))

        for t_ep in transition_eps:
            before = [c for ep, c in crit_series if t_ep - window * 3 <= ep < t_ep]
            if len(before) >= 2:
                trend = before[-1] - before[0]
                crit_trends_before.append(trend)

    if crit_trends_before:
        rising = sum(1 for t in crit_trends_before if t > 0)
        falling = sum(1 for t in crit_trends_before if t < 0)
        flat = sum(1 for t in crit_trends_before if t == 0)
        print(f"\n  顿悟前临界度趋势 (前{window*3}轮):")
        print(f"    上升: {rising} 次 | 下降: {falling} 次 | 持平: {flat} 次")
        print(f"    平均变化: {np.mean(crit_trends_before):+.1f}")

        # 统计检验: 顿悟时刻临界度是否显著高于非顿悟时刻
        if crit_at_non_transitions:
            from scipy import stats
            try:
                t_stat, p_val = stats.mannwhitneyu(crit_at_transitions, crit_at_non_transitions, alternative='greater')
                print(f"\n  Mann-Whitney U检验 (顿悟 > 非顿悟):")
                print(f"    U={t_stat:.0f}, p={p_val:.4f}")
            except Exception:
                # scipy不可用时用简单比较
                mean_diff = np.mean(crit_at_transitions) - np.mean(crit_at_non_transitions)
                print(f"\n  均值差: {mean_diff:+.1f}")

        print(f"\n  [ME-C4] 判定 (AI推断, 置信度: 中等):")
        if rising > falling * 1.5:
            print(f"    临界度在顿悟前倾向于上升 → 支持'临界度预测顿悟'假说")
        elif np.mean(crit_at_transitions) > np.mean(crit_at_non_transitions) * 1.5:
            print(f"    顿悟时刻临界度显著高于平时 → 临界度是有效的同步指标")
        else:
            print(f"    临界度与顿悟无明显关联 → 需要其他预测指标")
    else:
        print(f"\n  数据不足以判断趋势")


def analyze_raf_closure(all_episode_data):
    """分析6: RAF闭合检验 — 顿悟是否形成了自维持结构"""
    print("\n" + "=" * 60)
    print("分析6: RAF闭合检验 (自维持结构验证)")
    print("=" * 60)

    raf_true = 0
    raf_false = 0
    coverage_ratios = []

    for seed_data in all_episode_data:
        for ed in seed_data:
            if ed['has_phase_transition'] and ed['transition_detail']:
                td = ed['transition_detail']
                raf = td.get('raf_closure', {})
                if raf.get('is_raf', False):
                    raf_true += 1
                else:
                    raf_false += 1
                coverage_ratios.append(raf.get('coverage_ratio', 0))

    total = raf_true + raf_false
    print(f"  总顿悟次数: {total}")
    print(f"  通过RAF闭合检验: {raf_true} ({raf_true/max(total,1)*100:.0f}%)")
    print(f"  未通过RAF闭合检验: {raf_false} ({raf_false/max(total,1)*100:.0f}%)")

    if coverage_ratios:
        print(f"\n  覆盖率分布:")
        print(f"    平均: {np.mean(coverage_ratios):.3f}")
        print(f"    中位数: {np.median(coverage_ratios):.3f}")
        print(f"    范围: [{min(coverage_ratios):.3f}, {max(coverage_ratios):.3f}]")

        # 按覆盖率分段
        high = sum(1 for c in coverage_ratios if c >= 0.75)
        mid = sum(1 for c in coverage_ratios if 0.5 <= c < 0.75)
        low = sum(1 for c in coverage_ratios if c < 0.5)
        print(f"\n    高覆盖(≥0.75, 自维持): {high}")
        print(f"    中覆盖(0.5-0.75, 部分自维持): {mid}")
        print(f"    低覆盖(<0.5, 依赖外部): {low}")

    print(f"\n  [ME-C4] 判定 (AI推断, 置信度: 中等):")
    if total > 0 and raf_true / total > 0.5:
        print(f"    多数顿悟形成自维持结构 → RAF理论适用于我们的系统")
        print(f"    → 顿悟不仅是拓扑变化，更是功能性自组织")
    elif total > 0 and raf_true / total > 0.2:
        print(f"    部分顿悟形成自维持结构 → RAF是有效的筛选器")
        print(f"    → 可以用RAF区分'强顿悟'(自维持)和'弱顿悟'(非自维持)")
    else:
        print(f"    少数顿悟满足RAF闭合 → 可能需要调整RAF定义")
        print(f"    → 或者当前枢机集合的互连性不足")


def main():
    n_seeds = 5  # 用5个seed做详细分析（节省时间）
    n_episodes = 5000

    print("=" * 60)
    print("v1.2.2 顿悟深度分析 (含RAF闭合检验)")
    print(f"规模: {n_seeds} seeds × {n_episodes} 轮")
    print("=" * 60)

    all_episode_data = []
    for s in range(n_seeds):
        print(f"\n运行 seed {s}...", end=' ', flush=True)
        data = run_with_detailed_tracking(seed=s, n_episodes=n_episodes)
        n_pt = sum(1 for ed in data if ed['has_phase_transition'])
        print(f"完成 ({n_pt} 次顿悟)")
        all_episode_data.append(data)

    analyze_temporal_distribution(all_episode_data, n_episodes)
    analyze_behavioral_impact(all_episode_data, window=50)
    analyze_shuffled_baseline(all_episode_data, n_shuffles=50)
    analyze_pivot_content(all_episode_data)
    analyze_criticality_prediction(all_episode_data)
    analyze_raf_closure(all_episode_data)

    print("\n" + "=" * 60)
    print("分析完成")
    print("[ME-C4] 所有判定均为AI推断，置信度: 中等")
    print("=" * 60)


if __name__ == '__main__':
    main()
