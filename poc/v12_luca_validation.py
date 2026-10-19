"""
v1.2.2 LUCA理论验证实验

用TUF-E算法模型还原LUCA/原生细胞研究的三个核心结论:
1. RAF相变: 多样性超过临界阈值时自催化集突然涌现
2. 耗散维持: 能量通量低于阈值时结构崩溃
3. 三层耦合: 代谢+信息+边界缺一不可

[ME-C4] 所有判定均为AI推断，置信度: 中等
"""

import sys
import numpy as np
sys.path.insert(0, '.')
from tufe_grounding_v12 import (
    SharedEnvironment, MultiAgentSystem, ConsensusKnowledgeStore
)


def run_condition(seed, n_episodes, decay_rate=0.003, use_pivots=True,
                  use_consensus=True, track_cycles=True):
    """运行单个实验条件"""
    env = SharedEnvironment(seed=seed, feedback_delay=3)
    system = MultiAgentSystem(
        use_consensus=use_consensus,
        use_pivots=use_pivots,
        seed=seed
    )

    # 覆盖衰减率
    original_decay = decay_rate
    pending = {}
    total_correct = 0
    total_attempts = 0

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

            # 覆盖衰减率
            if use_consensus and use_pivots:
                system.consensus_store.decay(rate=original_decay)

    remaining = env.flush_feedback()
    if remaining:
        system.process_feedback(remaining, pending, n_episodes)

    confirmed = system.consensus_store.get_confirmed_pivots() if use_consensus else {}
    raf = system.phase_detector.check_raf_closure(confirmed) if use_pivots else {'is_raf': False, 'coverage_ratio': 0}
    stats = system.get_system_stats()

    return {
        'success_rate': total_correct / max(total_attempts, 1),
        'n_confirmed': len(confirmed),
        'feasible_volume': stats['feasible_space']['current_volume'],
        'raf_closure': raf,
        'n_phase_transitions': stats['phase_transitions']['n_phase_transitions'],
    }


def experiment1_raf_phase_transition():
    """验证1: RAF相变 — 枢机数量是否存在临界阈值"""
    print("=" * 60)
    print("验证1: RAF相变 (Kauffman/Steel)")
    print("假设: 枢机多样性超过临界阈值时，自维持结构突然涌现")
    print("=" * 60)

    # 通过改变episode数量来控制枢机积累量
    episode_levels = [100, 200, 500, 1000, 2000, 3000, 5000]
    n_seeds = 5

    results = []
    for n_ep in episode_levels:
        raf_count = 0
        avg_confirmed = 0
        avg_sr = 0
        for s in range(n_seeds):
            r = run_condition(seed=s, n_episodes=n_ep)
            if r['raf_closure']['is_raf']:
                raf_count += 1
            avg_confirmed += r['n_confirmed']
            avg_sr += r['success_rate']
        avg_confirmed /= n_seeds
        avg_sr /= n_seeds
        raf_rate = raf_count / n_seeds

        results.append({
            'episodes': n_ep,
            'avg_confirmed': avg_confirmed,
            'raf_rate': raf_rate,
            'avg_sr': avg_sr,
        })
        bar = "█" * int(raf_rate * 20)
        print(f"  {n_ep:5d}轮 | 平均枢机:{avg_confirmed:5.1f} | RAF达成率:{raf_rate:.0%} {bar} | SR:{avg_sr:.1%}")

    # 检测相变点
    raf_rates = [r['raf_rate'] for r in results]
    phase_found = False
    for i in range(1, len(raf_rates)):
        if raf_rates[i] - raf_rates[i-1] >= 0.4:  # 跳跃≥40%
            print(f"\n  [ME-C4] 检测到相变: {results[i-1]['episodes']}→{results[i]['episodes']}轮之间")
            print(f"    RAF达成率从 {raf_rates[i-1]:.0%} 跳到 {raf_rates[i]:.0%}")
            print(f"    对应枢机数量: {results[i-1]['avg_confirmed']:.1f} → {results[i]['avg_confirmed']:.1f}")
            phase_found = True
            break

    if not phase_found:
        if raf_rates[-1] > 0.5:
            print(f"\n  [ME-C4] RAF达成率随学习持续增长，未检测到尖锐相变")
            print(f"    可能是渐进式过渡，或临界点在更早期")
        else:
            print(f"\n  [ME-C4] 未达到足够的RAF达成率")

    print(f"\n  与LUCA研究对比:")
    print(f"    Kauffman RAF理论预测: 渗流型相变(突然涌现)")
    print(f"    我们的结果: {'支持相变假说' if phase_found else '渐进过渡或需更多数据'}")


def experiment2_dissipative_maintenance():
    """验证2: 耗散维持 — 能量通量与结构存续的关系"""
    print("\n" + "=" * 60)
    print("验证2: 耗散维持 (Prigogine/Lane&Martin)")
    print("假设: 衰减率过高(能量不足)时，可行空间崩溃")
    print("=" * 60)

    decay_rates = [0.0001, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    n_seeds = 5
    n_episodes = 3000

    results = []
    for decay in decay_rates:
        avg_vol = 0
        avg_sr = 0
        avg_confirmed = 0
        collapse_count = 0
        for s in range(n_seeds):
            r = run_condition(seed=s, n_episodes=n_episodes, decay_rate=decay)
            avg_vol += r['feasible_volume']
            avg_sr += r['success_rate']
            avg_confirmed += r['n_confirmed']
            if r['feasible_volume'] < 5:
                collapse_count += 1
        avg_vol /= n_seeds
        avg_sr /= n_seeds
        avg_confirmed /= n_seeds
        collapse_rate = collapse_count / n_seeds

        results.append({
            'decay': decay,
            'avg_vol': avg_vol,
            'avg_sr': avg_sr,
            'avg_confirmed': avg_confirmed,
            'collapse_rate': collapse_rate,
        })
        status = "💀崩溃" if collapse_rate > 0.5 else ("⚠️脆弱" if avg_vol < 20 else "✅稳定")
        print(f"  衰减率={decay:.4f} | 体积:{avg_vol:5.1f} | 枢机:{avg_confirmed:4.1f} | SR:{avg_sr:.1%} | 崩溃率:{collapse_rate:.0%} {status}")

    # 找临界衰减率
    stable_results = [r for r in results if r['collapse_rate'] <= 0.5]
    collapsed_results = [r for r in results if r['collapse_rate'] > 0.5]

    print(f"\n  [ME-C4] 判定:")
    if stable_results and collapsed_results:
        max_stable = max(r['decay'] for r in stable_results)
        min_collapsed = min(r['decay'] for r in collapsed_results)
        print(f"    稳定区间: 衰减率 ≤ {max_stable}")
        print(f"    崩溃区间: 衰减率 ≥ {min_collapsed}")
        print(f"    临界点约在 {max_stable}~{min_collapsed} 之间")
    elif all(r['collapse_rate'] <= 0.5 for r in results):
        print(f"    所有衰减率下均稳定 — 系统鲁棒性强")
    else:
        print(f"    所有衰减率下均崩溃 — 需要检查模型")

    print(f"\n  与LUCA研究对比:")
    print(f"    Lane&Martin: LUCA依赖~300mV质子梯度维持代谢")
    print(f"    Prigogine: 低于最低能量通量 → 耗散结构消失")
    print(f"    我们的结果: {'存在明确的能量阈值' if stable_results and collapsed_results else '需进一步分析'}")


def experiment3_three_layer_coupling():
    """验证3: 三层耦合 — 结构+时间+功能缺一不可"""
    print("\n" + "=" * 60)
    print("验证3: 三层耦合 (Gánti化学子)")
    print("假设: 缺失任一层，系统性能显著退化")
    print("=" * 60)

    n_seeds = 5
    n_episodes = 3000

    conditions = {
        '完整三层(A)': {'use_pivots': True, 'use_consensus': True},
        '无结构层(C)': {'use_pivots': False, 'use_consensus': False},
        '无共识(B)': {'use_pivots': True, 'use_consensus': False},
    }

    results = {}
    for name, params in conditions.items():
        avg_sr = 0
        avg_vol = 0
        avg_confirmed = 0
        avg_pt = 0
        for s in range(n_seeds):
            r = run_condition(seed=s, n_episodes=n_episodes, **params)
            avg_sr += r['success_rate']
            avg_vol += r['feasible_volume']
            avg_confirmed += r['n_confirmed']
            avg_pt += r['n_phase_transitions']
        avg_sr /= n_seeds
        avg_vol /= n_seeds
        avg_confirmed /= n_seeds
        avg_pt /= n_seeds

        results[name] = {
            'sr': avg_sr, 'vol': avg_vol,
            'confirmed': avg_confirmed, 'pt': avg_pt
        }
        print(f"  {name:12s} | SR:{avg_sr:.1%} | 体积:{avg_vol:5.1f} | 枢机:{avg_confirmed:4.1f} | 顿悟:{avg_pt:.1f}")

    # 比较
    full = results['完整三层(A)']
    print(f"\n  层级缺失的影响:")
    for name, data in results.items():
        if name == '完整三层(A)':
            continue
        sr_diff = data['sr'] - full['sr']
        vol_diff = data['vol'] - full['vol']
        pt_diff = data['pt'] - full['pt']
        print(f"    {name}: SR{sr_diff:+.1%}, 体积{vol_diff:+.1f}, 顿悟{pt_diff:+.1f}")

    print(f"\n  与Gánti化学子对比:")
    print(f"    Gánti: 代谢+信息+边界三者功能耦合，缺一不可")
    no_struct = results.get('无结构层(C)', {})
    no_consensus = results.get('无共识(B)', {})
    struct_loss = full['vol'] - no_struct.get('vol', 0)
    consensus_loss = full['vol'] - no_consensus.get('vol', 0)
    print(f"    缺失结构层损失: {struct_loss:.1f} 体积单位")
    print(f"    缺失共识层损失: {consensus_loss:.1f} 体积单位")
    if struct_loss > 0 and consensus_loss > 0:
        print(f"    → 两层都有不可替代的贡献，支持三层耦合假说")
    else:
        print(f"    → 某些层可能冗余，需要重新审视耦合关系")


def main():
    print("=" * 60)
    print("LUCA理论验证实验")
    print("用TUF-E算法模型还原原生细胞研究的核心结论")
    print("[ME-C4] 所有判定均为AI推断，置信度: 中等")
    print("=" * 60)

    experiment1_raf_phase_transition()
    experiment2_dissipative_maintenance()
    experiment3_three_layer_coupling()

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
