"""
v1.15 — skeleton-addressing GRANULARITY LADDER + PATH PROVENANCE

Implements EXACTLY poc/preregister/v15_granularity_provenance.yaml (committed
BEFORE any results, b9606eb). Reviewer ask #4 on a SINGLE-STEP domain
(arithmetic) where the SC-3 discovery bottleneck does NOT bind.

Arms (same task stream, same seeds, same cycles, ONE table):
  A0 random          : uniform random action
  A1 no-op           : always id
  A2 best-fixed      : best single fixed action = STATE-BLIND CEILING
  G0 bandit (shipped): TUFEAgent(exact_addressing=False)
  G1 exact           : TUFEAgent(exact_addressing=True, address_key_fn=None) [= v1.14 A4]
  G2 skel-64         : address_key_fn = 4-bin magnitude of (a,b,target) -> 64 groups
  G3 skel-8          : 2-bin magnitude of (a,b,target) -> 8 groups
  G4 skel-target-only: 4-bin magnitude of TARGET alone -> 4 groups

K-HUMAN-DOOR GUARD (critical): skeleton keys bin the MAGNITUDE (abs value) of the
observed quantities with fixed ROUND thresholds. They deliberately DROP the sign of
the target, because a negative target uniquely implies `sub` (a,b>0 => add/mul>0) and
would therefore ENCODE the op. Bins GROUP states; they do NOT say which action is
correct. The op-recovering structure must EMERGE from viability selection, and is
verified POST-HOC (never used as reward or viability). Viability is environment-defined
only: result == target.

RED LINE: fix the task and the metric, not the agent. ONE ladder, ONE seed set, ONE
run. If no interior granularity clears the ceiling, report H-EXACT-BEST / H-NOADV.
"""

import sys
import os
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # to import v11_state_dependent_pivot
sys.path.insert(0, os.path.dirname(_HERE))      # repo root, to import the tufe package

from tufe.engine import TUFEAgent
from v11_state_dependent_pivot import (
    ArithmeticObservableDomain, OPS, N_OPS, ID_ACTION,
    run_fixed_policy, state_conditional_headroom, _json_default,
)


# ============================================================
# FROZEN skeleton bin thresholds (generic round magnitudes, answer-agnostic)
# ============================================================
A_B_4BIN = [5, 10, 15]
A_B_2BIN = [10]
TARGET_4BIN = [10, 50, 150]
TARGET_2BIN = [30]


def _magbin(v, edges):
    """Generic magnitude bin: how many round edges |v| has reached. Sign is DROPPED
    on purpose (K-HUMAN-DOOR: sign of target would encode `sub`)."""
    av = abs(v)
    return sum(1 for t in edges if av >= t)


def _payload(raw_obs):
    p = raw_obs['magnitude']  # [a, b, target] for ArithmeticObservableDomain
    return p[0], p[1], p[2]


def skel64_key(raw_obs):
    a, b, tgt = _payload(raw_obs)
    return f"{_magbin(a, A_B_4BIN)}{_magbin(b, A_B_4BIN)}{_magbin(tgt, TARGET_4BIN)}"


def skel8_key(raw_obs):
    a, b, tgt = _payload(raw_obs)
    return f"{_magbin(a, A_B_2BIN)}{_magbin(b, A_B_2BIN)}{_magbin(tgt, TARGET_2BIN)}"


def skel_target_only_key(raw_obs):
    _, _, tgt = _payload(raw_obs)
    return f"{_magbin(tgt, TARGET_4BIN)}"


# exact (G1) uses address_key_fn=None -> full-state repr key built by the engine.


# ============================================================
# Agent runner (returns viability + fire stats + consolidated skeleton->action map)
# ============================================================
def run_agent(seed, cycles, exact_addressing, address_key_fn=None):
    dom = ArithmeticObservableDomain(seed=seed)
    agent = TUFEAgent(dom, seed=seed, exact_addressing=exact_addressing,
                      address_key_fn=address_key_fn)
    for _ in range(cycles):
        agent.run_cycle()
    vr = agent.viable_cycles / max(agent.total_cycles, 1)
    fires = agent.exact_fires
    fire_viable = agent.exact_fire_viable
    fire_rate = fires / max(cycles, 1)
    fire_viable_rate = (fire_viable / fires) if fires else None
    return {
        'viability_rate': float(vr),
        'confirmed_pivots': int(len(agent.knowledge.get_confirmed())),
        'feasible_volume': int(agent.feasible_space.stats()['current_volume']),
        'exact_fires': int(fires),
        'fire_rate': float(fire_rate),
        'fire_viable_rate': (float(fire_viable_rate) if fire_viable_rate is not None else None),
        '_agent': agent,
    }


def dominant_map(agent):
    """Consolidated skeleton -> dominant learned action (n>=3, max success ratio)."""
    m = {}
    for k, rec in agent.exact_mem.items():
        cands = [(v / n, n, aid) for aid, (n, v) in rec.items() if n >= 3]
        if cands:
            m[k] = int(max(cands)[2])
    return m


# ============================================================
# POST-HOC op-recovery reference (descriptive ruler ONLY — never reward/viability)
# ============================================================
def op_reference(key_fn, seed=999, n=200000):
    """For each skeleton group, the empirical distribution of CORRECT ops in the
    environment. This is the human/LLM-equivalent 'true relation' ruler. It is computed
    from the environment (result==target), NOT from any answer key."""
    dom = ArithmeticObservableDomain(seed=seed)
    group_ops = {}     # key -> np.array counts per op
    group_total = {}
    for _ in range(n):
        dom._gen()
        a, b, tgt = dom._task['a'], dom._task['b'], dom._task['target']
        raw = {'magnitude': [a, b, tgt]}
        k = key_fn(raw)
        if k not in group_ops:
            group_ops[k] = np.zeros(N_OPS)
            group_total[k] = 0
        group_total[k] += 1
        for aid, (_, fn) in OPS.items():
            try:
                if abs(fn(a, b) - tgt) < 1e-6:
                    group_ops[k][aid] += 1
            except Exception:
                pass
    ref = {}
    for k in group_ops:
        tot = group_total[k]
        counts = group_ops[k]
        if counts.sum() == 0:
            continue
        plur = int(np.argmax(counts))
        plur_frac = float(counts[plur] / tot)          # fraction of tasks in group
        plur_correct_frac = float(counts[plur] / max(counts.sum(), 1))
        ref[k] = {'plurality_op': plur, 'plurality_task_frac': plur_frac,
                  'plurality_of_correct': plur_correct_frac, 'n_tasks': tot}
    return ref


def classify_provenance(learned_map, ref, agreement_cut=0.70, determinate_cut=0.50):
    """Compare the agent's EMERGED skeleton->action map to the environment's true
    op relation. Descriptive classification (CONVERGENT/EQUIFINAL/DIVERGENT)."""
    populated = [k for k in learned_map if k in ref]
    determinate = [k for k in populated if ref[k]['plurality_of_correct'] >= determinate_cut]
    if not determinate:
        return {'class': 'INDETERMINATE', 'agreement': None,
                'n_populated': len(populated), 'n_determinate': 0}
    agree = sum(1 for k in determinate if learned_map[k] == ref[k]['plurality_op'])
    agreement = agree / len(determinate)
    if agreement >= agreement_cut:
        cls = 'CONVERGENT'
    elif agreement >= 0.30:
        cls = 'EQUIFINAL'   # partially recovers; different/mixed internal structure
    else:
        cls = 'DIVERGENT'
    return {'class': cls, 'agreement': float(agreement),
            'n_populated': len(populated), 'n_determinate': len(determinate),
            'n_agree': int(agree)}


def cross_seed_consistency(maps, agreement_cut=0.70):
    """maps: list of dominant_map dicts, one per seed, SAME key space.
    A group is 'consistent' if the same dominant action appears in >=4/5 seeds.
    Returns fraction of populated groups that are consistent."""
    all_keys = set()
    for m in maps:
        all_keys |= set(m.keys())
    populated = [k for k in all_keys if sum(1 for m in maps if k in m) >= 2]
    if not populated:
        return {'consistency': None, 'n_populated': 0}
    n_seeds = len(maps)
    need = int(np.ceil(0.8 * n_seeds))  # >=4 of 5
    consistent = 0
    for k in populated:
        acts = [m[k] for m in maps if k in m]
        if len(acts) < need:
            continue
        mode_act = max(set(acts), key=acts.count)
        if acts.count(mode_act) >= need:
            consistent += 1
    return {'consistency': float(consistent / len(populated)),
            'n_populated': len(populated), 'n_consistent': int(consistent),
            'need_seeds': need}


# ============================================================
def main():
    seeds = [42, 43, 44, 45, 46]
    cycles = 20000
    ADV_MARGIN, EXCESS_MARGIN, HEADROOM_MIN = 0.10, 0.05, 0.15
    CROSS_SEED_CUT = 0.70

    print("=" * 74)
    print("v1.15 SKELETON-ADDRESSING GRANULARITY LADDER + PATH PROVENANCE")
    print("=" * 74)

    # --- K-DISCRIMINATES: headroom re-asserted BEFORE any agent runs ---
    hr, oracle, bf = state_conditional_headroom(seeds[0], n=100000)
    print(f"\n[K-DISCRIMINATES] headroom={hr:.3f} (oracle={oracle:.3f}, best-fixed={bf:.3f})")
    print(f"  HEADROOM_MIN={HEADROOM_MIN} -> {'PASS' if hr >= HEADROOM_MIN else 'FAIL (VOID)'}")
    if hr < HEADROOM_MIN:
        print("  Task cannot separate a state-learner from a bandit. STOP.")
        return

    skeleton_arms = {
        'G0_bandit': dict(exact_addressing=False, address_key_fn=None),
        'G1_exact': dict(exact_addressing=True, address_key_fn=None),
        'G2_skel64': dict(exact_addressing=True, address_key_fn=skel64_key),
        'G3_skel8': dict(exact_addressing=True, address_key_fn=skel8_key),
        'G4_skel_target_only': dict(exact_addressing=True, address_key_fn=skel_target_only_key),
    }

    results = {}
    maps = {name: [] for name in skeleton_arms}  # per-seed dominant maps

    for seed in seeds:
        print(f"\n--- seed {seed} ({cycles} cycles) ---")
        a0 = run_fixed_policy(seed, cycles, lambda rng: int(rng.integers(0, N_OPS)))
        a1 = run_fixed_policy(seed, cycles, lambda rng: ID_ACTION)
        fixed_vrs = [run_fixed_policy(seed, cycles, lambda rng, aid=aid: aid)
                     for aid in range(N_OPS)]
        a2 = max(fixed_vrs)
        a2_best = int(np.argmax(fixed_vrs))

        arm_res = {}
        for name, kw in skeleton_arms.items():
            r = run_agent(seed, cycles, **kw)
            agent = r.pop('_agent')
            arm_res[name] = r
            if kw['exact_addressing']:
                maps[name].append(dominant_map(agent))

        results[seed] = {'A0_random': a0, 'A1_noop': a1, 'A2_best_fixed': a2,
                         'A2_best_action': a2_best, **arm_res}
        print(f"  A0 random           = {a0:.3f}")
        print(f"  A1 no-op (id)       = {a1:.3f}")
        print(f"  A2 best-fixed       = {a2:.3f}  (action {OPS[a2_best][0]})  <- CEILING")
        for name in skeleton_arms:
            r = arm_res[name]
            fv = r['fire_viable_rate']
            fvs = f"{fv:.3f}" if fv is not None else "  n/a"
            print(f"  {name:<20}= {r['viability_rate']:.3f}  fire_rate={r['fire_rate']:.3f} "
                  f"fire-viable={fvs}  V={r['feasible_volume']}")

    # ================= AGGREGATE =================
    print("\n" + "=" * 74)
    print("AGGREGATE (mean +/- sd over 5 seeds) — ALL ARMS IN ONE TABLE")
    print("=" * 74)

    def agg(getter):
        vals = [getter(results[s]) for s in seeds]
        return float(np.mean(vals)), float(np.std(vals))

    m_a0, _ = agg(lambda r: r['A0_random'])
    m_a1, _ = agg(lambda r: r['A1_noop'])
    m_a2, _ = agg(lambda r: r['A2_best_fixed'])
    table = {'A0_random': (m_a0, 0.0), 'A1_noop': (m_a1, 0.0), 'A2_best_fixed': (m_a2, 0.0)}
    for name in skeleton_arms:
        m, sd = agg(lambda r, name=name: r[name]['viability_rate'])
        table[name] = (m, sd)

    hdr = f"  {'arm':<22}{'viability':>16}{'fire_rate':>12}{'fire-viable':>14}{'vs A2':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, (m, sd) in table.items():
        fr = fv = None
        if name in skeleton_arms:
            frs = [results[s][name]['fire_rate'] for s in seeds]
            fvs = [results[s][name]['fire_viable_rate'] for s in seeds
                   if results[s][name]['fire_viable_rate'] is not None]
            fr = float(np.mean(frs))
            fv = float(np.mean(fvs)) if fvs else None
        fr_s = f"{fr:.3f}" if fr is not None else "  -"
        fv_s = f"{fv:.3f}" if fv is not None else "  -"
        d = m - m_a2
        star = " *" if d >= ADV_MARGIN else ""
        print(f"  {name:<22}{m:>10.3f}±{sd:<5.3f}{fr_s:>12}{fv_s:>14}{d:>+9.3f}{star}")
    print(f"\n  A2 state-blind CEILING = {m_a2:.3f}   ADV_MARGIN = {ADV_MARGIN}   (* = clears ceiling by margin)")

    # ---- P-COVERAGE / P-PRECISION-COST (monotone checks G1<G2<G3 fire-rate / fire-viable) ----
    def mean_fire(name, key):
        vals = [results[s][name][key] for s in seeds
                if results[s][name][key] is not None]
        return float(np.mean(vals)) if vals else None

    fr_g1 = mean_fire('G1_exact', 'fire_rate')
    fr_g2 = mean_fire('G2_skel64', 'fire_rate')
    fr_g3 = mean_fire('G3_skel8', 'fire_rate')
    fv_g1 = mean_fire('G1_exact', 'fire_viable_rate')
    fv_g2 = mean_fire('G2_skel64', 'fire_viable_rate')
    fv_g3 = mean_fire('G3_skel8', 'fire_viable_rate')
    p_coverage = (fr_g1 is not None and fr_g2 is not None and fr_g3 is not None
                  and fr_g1 < fr_g2 < fr_g3)
    p_precision_cost = (fv_g1 is not None and fv_g2 is not None and fv_g3 is not None
                        and fv_g1 > fv_g2 > fv_g3)
    print(f"\n  P-COVERAGE (fire-rate G1<G2<G3): {fr_g1:.3f}<{fr_g2:.3f}<{fr_g3:.3f} -> "
          f"{'PASS' if p_coverage else 'FAIL'}")
    print(f"  P-PRECISION-COST (fire-viable G1>G2>G3): {fv_g1:.3f}>{fv_g2:.3f}>{fv_g3:.3f} -> "
          f"{'PASS' if p_precision_cost else 'FAIL'}")

    # ---- P-INTERIOR-OPTIMUM: exists G* in {G2,G3,G4} beating A2 by margin AND G1 AND G0 ----
    interior = ['G2_skel64', 'G3_skel8', 'G4_skel_target_only']
    m_g0 = table['G0_bandit'][0]
    m_g1 = table['G1_exact'][0]
    gstar = max(interior, key=lambda n: table[n][0])
    m_gstar = table[gstar][0]
    p_interior = ((m_gstar - m_a2) >= ADV_MARGIN and m_gstar > m_g1 and m_gstar > m_g0)
    print(f"\n  P-INTERIOR-OPTIMUM: G* = {gstar} (viability {m_gstar:.3f})")
    print(f"    beats A2 ceiling by >= {ADV_MARGIN}: {(m_gstar - m_a2):+.3f} -> "
          f"{'YES' if (m_gstar - m_a2) >= ADV_MARGIN else 'NO'}")
    print(f"    beats G1 exact ({m_g1:.3f}): {'YES' if m_gstar > m_g1 else 'NO'}")
    print(f"    beats G0 bandit ({m_g0:.3f}): {'YES' if m_gstar > m_g0 else 'NO'}")
    print(f"    -> P-INTERIOR-OPTIMUM {'PASS' if p_interior else 'FAIL'}")

    # ---- P-EXCESS (OC-9): G* fire-viable - background rho (A0) ----
    fv_gstar = mean_fire(gstar, 'fire_viable_rate')
    excess = (fv_gstar - m_a0) if fv_gstar is not None else None
    p_excess = (excess is not None and excess >= EXCESS_MARGIN)
    if excess is not None:
        print(f"\n  P-EXCESS (OC-9): G* fire-viable {fv_gstar:.3f} - rho(A0) {m_a0:.3f} "
              f"= {excess:+.3f} -> {'PASS' if p_excess else 'FAIL'} (min {EXCESS_MARGIN})")

    # ================= PATH PROVENANCE =================
    print("\n" + "=" * 74)
    print(f"PATH PROVENANCE for G* = {gstar} (instruments; post-hoc ruler, NEVER reward)")
    print("=" * 74)
    gstar_keyfn = skeleton_arms[gstar]['address_key_fn']

    # (a) cold-start: architectural — asserted (empty table at t=0, no op/answer channel)
    print("  (a) COLD-START: empty address table at t=0, no op channel, no answer channel.")
    print("      viability = environment (result==target) only. -> guarantee holds by construction.")

    # (b) cross-seed re-derivation
    xs = cross_seed_consistency(maps[gstar])
    p_cross = (xs['consistency'] is not None and xs['consistency'] >= CROSS_SEED_CUT)
    print(f"  (b) CROSS-SEED RE-DERIVATION: consistency={xs['consistency']} "
          f"({xs.get('n_consistent')}/{xs['n_populated']} populated groups agree in "
          f">={xs.get('need_seeds')}/5 seeds)")
    print(f"      cut {CROSS_SEED_CUT} -> {'PASS (environmental attractor)' if p_cross else 'FAIL (per-seed fluke)'}")

    # (c) post-hoc op-recovery
    ref = op_reference(gstar_keyfn)
    prov = classify_provenance(maps[gstar][0], ref)  # seed-0 map vs env true relation
    # also aggregate across seeds: use the modal map per group
    modal_map = {}
    all_keys = set()
    for m in maps[gstar]:
        all_keys |= set(m.keys())
    for k in all_keys:
        acts = [m[k] for m in maps[gstar] if k in m]
        if acts:
            modal_map[k] = max(set(acts), key=acts.count)
    prov_modal = classify_provenance(modal_map, ref)
    print(f"  (c) POST-HOC OP-RECOVERY (seed-0 map): class={prov['class']} "
          f"agreement={prov['agreement']} ({prov.get('n_agree')}/{prov['n_determinate']} determinate groups)")
    print(f"      POST-HOC OP-RECOVERY (cross-seed modal map): class={prov_modal['class']} "
          f"agreement={prov_modal['agreement']} "
          f"({prov_modal.get('n_agree')}/{prov_modal['n_determinate']} determinate groups)")
    print("      NOTE: descriptive ruler only. Not used for viability, reward, or selection.")
    print("      HONEST LIMIT: arithmetic is single-step -> 'path' = one action -> provenance thin.")

    # ================= JUDGMENT =================
    print("\n" + "=" * 74)
    print("JUDGMENT (decided BEFORE seeing numbers, per prereg)")
    print("=" * 74)
    if p_interior and p_excess:
        verdict = 'H-OPTIMUM'
    elif m_g1 >= max(table[n][0] for n in interior):
        verdict = 'H-EXACT-BEST'
    elif (gstar == 'G4_skel_target_only' and m_gstar >= m_g3 >= m_g2 >= m_g1
          and (m_gstar - m_a2) >= ADV_MARGIN):
        verdict = 'H-COARSE-MONOTONE (SUSPICIOUS -> K-HUMAN-DOOR audit before claiming)'
    elif (m_gstar - m_a2) < ADV_MARGIN:
        verdict = 'H-NOADV'
    else:
        verdict = 'H-NOADV'
    print(f"  VERDICT: {verdict}")
    if verdict.startswith('H-COARSE-MONOTONE'):
        print("  AUDIT: G4 keys bin TARGET MAGNITUDE only, sign dropped, no op encoding.")
        print("         If op-recovery class is CONVERGENT via EMERGENCE (not the key), report;")
        print("         re-inspect skel_target_only_key for answer leakage before any claim.")

    out = {
        'study': 'v15_skeleton_granularity_ladder_and_provenance',
        'seeds': seeds, 'cycles': cycles,
        'headroom_seed0': hr,
        'aggregate': {k: {'mean': v[0], 'sd': v[1]} for k, v in table.items()},
        'fire_stats': {name: {'fire_rate': mean_fire(name, 'fire_rate'),
                              'fire_viable_rate': mean_fire(name, 'fire_viable_rate')}
                       for name in skeleton_arms if skeleton_arms[name]['exact_addressing']},
        'predictions': {
            'P_COVERAGE': bool(p_coverage), 'P_PRECISION_COST': bool(p_precision_cost),
            'P_INTERIOR_OPTIMUM': bool(p_interior), 'P_EXCESS': bool(p_excess),
            'gstar': gstar, 'gstar_viability': m_gstar,
            'oc9_excess': excess,
        },
        'provenance': {
            'cold_start': 'guaranteed_by_construction',
            'cross_seed': xs, 'P_CROSS_SEED': bool(p_cross),
            'op_recovery_seed0': prov, 'op_recovery_modal': prov_modal,
            'honest_limit': 'arithmetic_single_step_path_thin',
        },
        'verdict': verdict,
        'per_seed': {str(s): {k: (v if not isinstance(v, dict) else
                                  {kk: vv for kk, vv in v.items()})
                              for k, v in results[s].items()} for s in seeds},
    }
    with open('poc/v15_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    print("\n  wrote poc/v15_results.json")


if __name__ == '__main__':
    main()
