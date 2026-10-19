"""
v1.2.3 参数扫描实验

系统扫描三个核心参数的组合空间:
1. 衰减率 (decay_rate): 知识耗散速度
2. 共识阈值 (min_observers): 枢机确认所需的独立观察者数
3. 窗口大小 (window_size): 可行空间趋势计算的滑动窗口

目标: 确定系统的工作区间、最优区域和崩溃边界。

[ME-C4] 所有判定均为AI推断，置信度: 中等
"""

import sys
import numpy as np
sys.path.insert(0, '.')
from tufe_grounding_v12 import (
    SharedEnvironment, MultiAgentSystem, ConsensusKnowledgeStore
)


def run_one(seed, n_episodes, decay_rate, min_observers, window_size):
    """运行单个参数组合"""
    env = SharedEnvironment(seed=seed, feedback_delay=3)
    system = MultiAgentSystem(use_consensus=True, use_pivots=True, seed=seed)

    # 覆盖参数
    system.consensus_store.min_observers = min_observers
    system.feasible_tracker.window_size = window_size

    pending = {}
    total_correct = 0
    total_attempts = 0
    n_transitions = 0

    for ep in range(n_episodes):
        r = np.random.default_rng(ep + seed).random()
        difficulty = 1 if r < 0.5 else (2 if r < 0.8 else 3)
        task = env.generate_task(difficulty)
        pending[ep] = task

        fb_batch, agent_idx = system.run_episode(env, task, ep, pending)
        if fb_batch:
            learn_result = system.process_feedback(fb_batch, pending, ep)
            for fb in fb_batch:
                total_attempts += 1
                if fb.get('correct', False):
                    total_correct += 1
                pending.pop(fb['episode'], None)

            system.consensus_store.decay(rate=decay_rate)

            if learn_result and learn_result.get('phase_transition'):
                n_transitions += 1

    remaining = env.flush_feedback()
    if remaining:
        system.process_feedback(remaining, pending, n_episodes)

    confirmed = system.consensus_store.get_confirmed_pivots()
    raf = system.phase_detector.check_raf_closure(confirmed)
    stats = system.get_system_stats()

    return {
        'success_rate': total_correct / max(total_attempts, 1),
        'n_confirmed': len(confirmed),
        'feasible_volume': stats['feasible_space']['current_volume'],
        'raf_closed': raf['is_raf'],
        'n_transitions': n_transitions,
    }


def main():
    # 参数网格
    decay_rates = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    min_obs_list = [2, 3]  # 3个观察者中取2或3
    window_sizes = [10, 20, 50]

    n_seeds = 3
    n_episodes = 3000

    print("=" * 70)
    print("v1.2.3 参数扫描")
    print(f"规模: {len(decay_rates)}×{len(min_obs_list)}×{len(window_sizes)} = "
          f"{len(decay_rates)*len(min_obs_list)*len(window_sizes)} 组合 × {n_seeds} seeds × {n_episodes}轮")
    print("=" * 70)

    all_results = []

    for decay in decay_rates:
        for min_obs in min_obs_list:
            for win in window_sizes:
                avg_sr = 0
                avg_vol = 0
                avg_conf = 0
                avg_pt = 0
                raf_count = 0

                for s in range(n_seeds):
                    r = run_one(s, n_episodes, decay, min_obs, win)
                    avg_sr += r['success_rate']
                    avg_vol += r['feasible_volume']
                    avg_conf += r['n_confirmed']
                    avg_pt += r['n_transitions']
                    if r['raf_closed']:
                        raf_count += 1

                avg_sr /= n_seeds
                avg_vol /= n_seeds
                avg_conf /= n_seeds
                avg_pt /= n_seeds
                raf_rate = raf_count / n_seeds

                result = {
                    'decay': decay, 'min_obs': min_obs, 'window': win,
                    'sr': avg_sr, 'vol': avg_vol, 'conf': avg_conf,
                    'pt': avg_pt, 'raf': raf_rate
                }
                all_results.append(result)

                # 状态标记
                if avg_vol < 5 or avg_conf < 1:
                    status = "💀"
                elif raf_rate >= 0.6 and avg_sr >= 0.18:
                    status = "★"
                elif raf_rate >= 0.3:
                    status = "○"
                else:
                    status = "·"

                print(f"  {status} d={decay:.4f} obs={min_obs} w={win:2d} | "
                      f"SR:{avg_sr:.1%} 体积:{avg_vol:5.1f} 枢机:{avg_conf:4.1f} "
                      f"顿悟:{avg_pt:5.1f} RAF:{raf_rate:.0%}")

    # === 分析 ===
    print("\n" + "=" * 70)
    print("分析结果")
    print("=" * 70)

    # 1. 找最优组合 (综合评分: SR + 体积归一化 + RAF)
    for r in all_results:
        r['score'] = r['sr'] * 100 + (r['vol'] / 200) * 20 + r['raf'] * 30 + (r['conf'] / 30) * 10

    ranked = sorted(all_results, key=lambda x: x['score'], reverse=True)
    print("\n【TOP-5 最优参数组合】")
    for i, r in enumerate(ranked[:5]):
        print(f"  {i+1}. d={r['decay']:.4f} obs={r['min_obs']} w={r['window']:2d} | "
              f"SR:{r['sr']:.1%} 体积:{r['vol']:.0f} 枢机:{r['conf']:.0f} "
              f"RAF:{r['raf']:.0%} 综合分:{r['score']:.1f}")

    # 2. 找崩溃边界
    print("\n【崩溃边界】(体积<5 或 枢机<1)")
    collapse_found = False
    for r in all_results:
        if r['vol'] < 5 or r['conf'] < 1:
            print(f"  d={r['decay']:.4f} obs={r['min_obs']} w={r['window']:2d} | "
                  f"体积:{r['vol']:.1f} 枢机:{r['conf']:.1f}")
            collapse_found = True
    if not collapse_found:
        print("  未观测到崩溃 — 系统在扫描范围内鲁棒")

    # 3. 各参数的边际效应
    print("\n【参数敏感度分析】")

    # 衰减率效应 (固定其他参数为中位数)
    print("\n  衰减率效应 (obs=2, w=20):")
    for decay in decay_rates:
        match = [r for r in all_results if r['decay'] == decay and r['min_obs'] == 2 and r['window'] == 20]
        if match:
            r = match[0]
            bar = "█" * int(r['sr'] * 100)
            print(f"    d={decay:.4f} | SR:{r['sr']:.1%} {bar} | 枢机:{r['conf']:.0f} RAF:{r['raf']:.0%}")

    # 共识阈值效应
    print("\n  共识阈值效应 (d=0.003, w=20):")
    for mo in min_obs_list:
        match = [r for r in all_results if r['decay'] == 0.003 and r['min_obs'] == mo and r['window'] == 20]
        if match:
            r = match[0]
            print(f"    obs={mo} | SR:{r['sr']:.1%} 体积:{r['vol']:.0f} 枢机:{r['conf']:.0f} RAF:{r['raf']:.0%}")

    # 窗口大小效应
    print("\n  窗口大小效应 (d=0.003, obs=2):")
    for w in window_sizes:
        match = [r for r in all_results if r['decay'] == 0.003 and r['min_obs'] == 2 and r['window'] == w]
        if match:
            r = match[0]
            print(f"    w={w:2d} | SR:{r['sr']:.1%} 体积:{r['vol']:.0f} 枢机:{r['conf']:.0f} RAF:{r['raf']:.0%}")

    # 4. 工作区间总结
    print("\n" + "=" * 70)
    print("【工作区间总结】")
    good = [r for r in all_results if r['raf'] >= 0.6 and r['sr'] >= 0.15]
    if good:
        decays_good = [r['decay'] for r in good]
        print(f"  有效工作区: 衰减率 {min(decays_good):.4f}~{max(decays_good):.4f}")
        print(f"  共 {len(good)}/{len(all_results)} 个组合达到有效标准 (RAF≥60%, SR≥15%)")
    else:
        print("  无组合同时满足 RAF≥60% 且 SR≥15%")

    stable = [r for r in all_results if r['vol'] >= 50 and r['conf'] >= 5]
    print(f"  稳定区: {len(stable)}/{len(all_results)} 个组合 (体积≥50, 枢机≥5)")

    print("\n" + "=" * 70)
    print("扫描完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
