"""
v1.1 — does a STATE-DEPENDENT pivot learner beat a state-blind bandit?

Implements EXACTLY poc/preregister/v11_state_dependent_pivot.yaml (committed
before any results). Reviewer ask #4 is the credibility bar.

Arms (same seeds, same task stream, one table):
  A0 random          : uniform random action
  A1 no-op           : always the identity action (id)
  A2 best-fixed      : best single fixed action = state-blind ceiling
  A3 bandit (shipped): TUFEAgent(exact_addressing=False)  [released v1.0 engine]
  A4 state-dependent : TUFEAgent(exact_addressing=True)   [state-conditional pivots]
  frozen_random      : background rho for OC-9 excess causal validity

RED LINE: we fix the TASK (expose the target = a sensing channel, not an oracle)
and the METRIC (report every number beside its baseline). We do NOT tune the
agent to win. If A4 <= A2, we report H-NOADV honestly.
"""

import sys
import json
import numpy as np

from tufe.domain import DomainSpec, ObserverSpec, ActionResult
from tufe.engine import TUFEAgent


# ============================================================
# TASK: ArithmeticObservableDomain — target EXPOSED (reviewer ask #1)
# ============================================================
# The released ArithmeticDomain never shows the target -> viability is chance-capped
# ~1/3 and a fixed "always-add" matches the agent. Here observe() returns operands AND
# target, so the correct op is a deterministic function of the observed state. The agent
# is NOT told which op is correct; it must discover the (operands,target)->op coupling.
OPS = {
    0: ('add', lambda a, b: a + b),
    1: ('sub', lambda a, b: a - b),
    2: ('mul', lambda a, b: a * b),
    3: ('div', lambda a, b: a / b if b != 0 else float('inf')),
    4: ('neg', lambda a, b: -a),
    5: ('abs', lambda a, b: abs(a)),
    6: ('sqr', lambda a, b: a * a),
    7: ('id',  lambda a, b: a),
}
N_OPS = len(OPS)
ID_ACTION = 7


class _ObsMag(ObserverSpec):
    def __init__(self):
        super().__init__(name='magnitude', description='operand magnitude + sign', n_buckets=16)

    def discretize(self, raw):
        val = raw[0] if isinstance(raw, (list, tuple)) and raw else (raw or 0)
        av = abs(val)
        sign_bit = 0 if val >= 0 else 1
        thresholds = [3, 6, 10, 15, 25, 40, 60, 100]
        level = sum(1 for t in thresholds if av >= t)
        return f'b{min(level + sign_bit * 8, 15):02d}'


class _ObsSym(ObserverSpec):
    def __init__(self):
        super().__init__(name='symbolic', description='parity + sign + size', n_buckets=16)

    def discretize(self, raw):
        val = raw[0] if isinstance(raw, (list, tuple)) and raw else (raw or 0)
        parity = int(val) % 2
        sign_bit = 0 if val >= 0 else 1
        av = abs(val)
        size = 0 if av < 5 else (1 if av < 15 else (2 if av < 50 else 3))
        return f'b{min(size * 4 + sign_bit * 2 + parity, 15):02d}'


class _ObsTgt(ObserverSpec):
    def __init__(self, rng=None):
        super().__init__(name='targetmag', description='target magnitude', n_buckets=16)
        self._rng = rng or np.random.default_rng(0)

    def discretize(self, raw):
        # raw = [a, b, target]; this observer sees the TARGET magnitude (coarse).
        tgt = raw[2] if isinstance(raw, (list, tuple)) and len(raw) >= 3 else 0
        av = abs(tgt)
        thresholds = [3, 6, 10, 15, 25, 40, 60, 100, 200, 400]
        level = sum(1 for t in thresholds if av >= t)
        sign_bit = 0 if tgt >= 0 else 1
        return f'b{min(level + sign_bit * 11, 15):02d}'


class ArithmeticObservableDomain(DomainSpec):
    """Same generative rule as ArithmeticDomain, but the target is OBSERVABLE.

    GC-5 viability = |result - target| < eps (self-maintenance of problem-solving
    capability), NOT a reward to maximize. Task stream depends only on self.rng, so
    every arm sees an identical sequence of tasks for a given seed.
    """

    def __init__(self, seed=42, op_pool=(0, 1, 2)):
        self.rng = np.random.default_rng(seed)
        self.op_pool = list(op_pool)
        self._task = None
        self._obs_tgt = _ObsTgt(rng=self.rng)

    def name(self):
        return "ArithmeticObservable"

    def observers(self):
        return [_ObsMag(), _ObsSym(), self._obs_tgt]

    def observe(self):
        if self._task is None:
            self._gen()
        payload = [self._task['a'], self._task['b'], self._task['target']]
        return {'magnitude': payload, 'symbolic': payload, 'targetmag': payload}

    def action_space_size(self):
        return N_OPS

    def act(self, action_id):
        if self._task is None:
            self._gen()
        t = self._task
        _, fn = OPS.get(action_id, OPS[ID_ACTION])
        try:
            result = fn(t['a'], t['b'])
            correct = abs(result - t['target']) < 1e-6
        except Exception:
            result, correct = None, False
        self._gen()  # next task
        return ActionResult(
            success=correct,
            post_state={'a': self._task['a'], 'b': self._task['b'],
                        'target': self._task['target'], 'correct': correct},
            metadata={'correct': correct},
        )

    def assess_viability(self, state):
        return bool(state.get('correct', False))

    def apply_dissipation(self):
        pass

    def meta_rules(self):
        return {'op_pool': self.op_pool, 'target_observable': True}

    def _gen(self):
        a = int(self.rng.integers(1, 20))
        b = int(self.rng.integers(1, 20))
        op_id = int(self.rng.choice(self.op_pool))
        target = OPS[op_id][1](a, b)
        self._task = {'a': a, 'b': b, 'op': op_id, 'target': target}


# ============================================================
# P-HEADROOM: task-design check (computed BEFORE any agent runs)
# ============================================================
def state_conditional_headroom(seed, n=200000):
    """headroom = oracle state-conditional viability - best fixed-action viability.

    The oracle here is a TASK-DESIGN diagnostic only (K-DISCRIMINATES): it is never an
    agent and never reported as TUF-E performance. It answers 'can this task separate a
    state-learner from a bandit at all?'
    """
    dom = ArithmeticObservableDomain(seed=seed)
    per_action_hits = np.zeros(N_OPS)
    oracle_hits = 0
    for _ in range(n):
        dom._gen()
        a, b, tgt = dom._task['a'], dom._task['b'], dom._task['target']
        any_match = False
        for aid, (_, fn) in OPS.items():
            try:
                if abs(fn(a, b) - tgt) < 1e-6:
                    per_action_hits[aid] += 1
                    any_match = True
            except Exception:
                pass
        oracle_hits += int(any_match)
    best_fixed = per_action_hits.max() / n
    oracle = oracle_hits / n
    return float(oracle - best_fixed), float(oracle), float(best_fixed)


# ============================================================
# Arms
# ============================================================
def run_fixed_policy(seed, cycles, action_fn):
    """A0/A1/A2: a state-blind fixed (or random) policy on the same task stream."""
    dom = ArithmeticObservableDomain(seed=seed)
    rng = np.random.default_rng(1000 + seed)
    viable = 0
    for _ in range(cycles):
        aid = action_fn(rng)
        r = dom.act(aid)
        viable += int(dom.assess_viability(r.post_state))
    return viable / cycles


def run_agent(seed, cycles, exact_addressing):
    """A3 (bandit, exact_addressing=False) / A4 (state-dependent, exact_addressing=True)."""
    dom = ArithmeticObservableDomain(seed=seed)
    agent = TUFEAgent(dom, seed=seed, exact_addressing=exact_addressing)
    for _ in range(cycles):
        agent.run_cycle()
    vr = agent.viable_cycles / max(agent.total_cycles, 1)
    fires = agent.exact_fires
    fire_viable = agent.exact_fire_viable
    fire_rate = (fire_viable / fires) if fires else None
    return {
        'viability_rate': float(vr),
        'confirmed_pivots': int(len(agent.knowledge.get_confirmed())),
        'feasible_volume': int(agent.feasible_space.stats()['current_volume']),
        'exact_fires': int(fires),
        'fire_viable_rate': (float(fire_rate) if fire_rate is not None else None),
    }


def feasible_trajectory(seed, cycles, sample_every=1000):
    """P-F1: A4 feasible-space volume trajectory + saturation time."""
    dom = ArithmeticObservableDomain(seed=seed)
    agent = TUFEAgent(dom, seed=seed, exact_addressing=True)
    traj = []
    for c in range(cycles):
        agent.run_cycle()
        if (c + 1) % sample_every == 0:
            traj.append(agent.feasible_space.stats()['current_volume'])
    # saturation time = first sample index after which volume never rises >5%
    sat = None
    for i in range(len(traj)):
        if all(v <= traj[i] * 1.05 for v in traj[i:]):
            sat = (i + 1) * sample_every
            break
    return traj, sat


def main():
    seeds = [42, 43, 44, 45, 46]
    cycles = 20000
    HEADROOM_MIN, ADV_MARGIN, EXCESS_MARGIN = 0.15, 0.10, 0.05

    print("=" * 70)
    print("v1.1 STATE-DEPENDENT PIVOT vs STATE-BLIND BANDIT")
    print("=" * 70)

    # --- P-HEADROOM (task design check, before any agent) ---
    print("\n[P-HEADROOM] task-design discriminability (oracle diagnostic, NOT an agent):")
    hr_seed0, oracle0, bf0 = state_conditional_headroom(seeds[0], n=100000)
    print(f"  seed {seeds[0]}: headroom={hr_seed0:.3f} "
          f"(oracle state-conditional={oracle0:.3f}, best fixed={bf0:.3f})")
    p_headroom = hr_seed0 >= HEADROOM_MIN
    print(f"  HEADROOM_MIN={HEADROOM_MIN} -> P-HEADROOM {'PASS' if p_headroom else 'FAIL (VOID, K-DISCRIMINATES)'}")
    if not p_headroom:
        print("  Task cannot separate a state-learner from a bandit. STOP. Redesign task.")
        return

    results = {}
    for seed in seeds:
        print(f"\n--- seed {seed} ({cycles} cycles) ---")
        a0 = run_fixed_policy(seed, cycles, lambda rng: int(rng.integers(0, N_OPS)))
        a1 = run_fixed_policy(seed, cycles, lambda rng: ID_ACTION)
        # A2 best-fixed: pick the best fixed action empirically (honest state-blind ceiling)
        fixed_vrs = []
        for aid in range(N_OPS):
            fixed_vrs.append(run_fixed_policy(seed, cycles, lambda rng, aid=aid: aid))
        a2 = max(fixed_vrs)
        a2_best = int(np.argmax(fixed_vrs))
        a3 = run_agent(seed, cycles, exact_addressing=False)
        a4 = run_agent(seed, cycles, exact_addressing=True)
        results[seed] = {'A0_random': a0, 'A1_noop': a1, 'A2_best_fixed': a2,
                         'A2_best_action': a2_best,
                         'A3_bandit': a3, 'A4_state_dependent': a4}
        print(f"  A0 random          = {a0:.3f}")
        print(f"  A1 no-op (id)      = {a1:.3f}")
        print(f"  A2 best-fixed      = {a2:.3f}  (action {OPS[a2_best][0]})")
        print(f"  A3 bandit (shipped)= {a3['viability_rate']:.3f}  pivots={a3['confirmed_pivots']}")
        print(f"  A4 state-dependent = {a4['viability_rate']:.3f}  pivots={a4['confirmed_pivots']} "
              f"fires={a4['exact_fires']} fire-viable={a4['fire_viable_rate']}")

    # --- P-F1: feasible-space saturation on A4 (seed 0) ---
    print("\n[P-F1] A4 feasible-space volume trajectory (saturation check):")
    traj, sat = feasible_trajectory(seeds[0], cycles)
    print(f"  V(t) @1k steps: {traj}")
    print(f"  saturation time: {sat}  (None = still growing at horizon)")
    results['F1_trajectory'] = {'traj': traj, 'saturation_time': sat}

    # --- Aggregate + judgments ---
    print("\n" + "=" * 70)
    print("AGGREGATE (mean over 5 seeds)")
    print("=" * 70)
    m_a0 = np.mean([results[s]['A0_random'] for s in seeds])
    m_a1 = np.mean([results[s]['A1_noop'] for s in seeds])
    m_a2 = np.mean([results[s]['A2_best_fixed'] for s in seeds])
    m_a3 = np.mean([results[s]['A3_bandit']['viability_rate'] for s in seeds])
    m_a4 = np.mean([results[s]['A4_state_dependent']['viability_rate'] for s in seeds])
    sd_a4 = np.std([results[s]['A4_state_dependent']['viability_rate'] for s in seeds])

    # OC-9 excess: A4 fire-viable rate - background rho (random = A0)
    fire_rates = [results[s]['A4_state_dependent']['fire_viable_rate'] for s in seeds
                  if results[s]['A4_state_dependent']['fire_viable_rate'] is not None]
    m_fire = np.mean(fire_rates) if fire_rates else None
    excess = (m_fire - m_a0) if m_fire is not None else None

    print(f"  A0 random           = {m_a0:.3f}")
    print(f"  A1 no-op            = {m_a1:.3f}")
    print(f"  A2 best-fixed       = {m_a2:.3f}   <- state-blind CEILING")
    print(f"  A3 bandit (shipped) = {m_a3:.3f}")
    print(f"  A4 state-dependent  = {m_a4:.3f} ± {sd_a4:.3f}")
    if excess is not None:
        print(f"  A4 fire-viable      = {m_fire:.3f} | OC-9 excess (rho=A0 {m_a0:.3f}) = {excess:.3f}")

    p_bandit_ceiling = (m_a4 - m_a2) >= ADV_MARGIN
    p_all = ((m_a4 - m_a0) >= ADV_MARGIN and (m_a4 - m_a1) >= ADV_MARGIN and
             (m_a4 - m_a2) >= ADV_MARGIN and (m_a4 - m_a3) >= ADV_MARGIN)
    p_excess = (excess is not None and excess >= EXCESS_MARGIN)

    print(f"\n  P-BANDIT-CEILING (A4-A2 >= {ADV_MARGIN}): {'PASS' if p_bandit_ceiling else 'FAIL'}")
    print(f"  P-ALL-BASELINES  (A4 beats all by {ADV_MARGIN}): {'PASS' if p_all else 'FAIL'}")
    print(f"  P-EXCESS (OC-9 >= {EXCESS_MARGIN}): {'PASS' if p_excess else 'FAIL'}")

    if p_bandit_ceiling and p_all and p_excess:
        verdict = 'H-PIVOT-WINS'
    elif p_bandit_ceiling:
        verdict = 'H-PARTIAL'
    else:
        verdict = 'H-NOADV'
    print(f"\n  VERDICT: {verdict}")

    out = {
        'study': 'v11_state_dependent_pivot_vs_bandit',
        'seeds': seeds, 'cycles': cycles,
        'headroom_seed0': hr_seed0,
        'aggregate': {'A0_random': m_a0, 'A1_noop': m_a1, 'A2_best_fixed': m_a2,
                      'A3_bandit': m_a3, 'A4_state_dependent': m_a4, 'A4_sd': sd_a4,
                      'A4_fire_viable': m_fire, 'oc9_excess': excess},
        'per_seed': {str(s): results[s] for s in seeds},
        'F1': results['F1_trajectory'],
        'flags': {'P_HEADROOM': bool(p_headroom), 'P_BANDIT_CEILING': bool(p_bandit_ceiling),
                  'P_ALL_BASELINES': bool(p_all), 'P_EXCESS': bool(p_excess)},
        'verdict': verdict,
    }
    with open('poc/v11_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)
    print("\n  wrote poc/v11_results.json")


def _json_default(o):
    """Cast numpy scalars to native Python for JSON (output-serializer fix only)."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f'not JSON serializable: {type(o)}')


if __name__ == '__main__':
    main()
