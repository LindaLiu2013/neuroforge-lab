"""v1.4 M6 verification: RG coarse-graining fixed-point plateau.

Distilled model M6 (renormalization group): the maintenance policy pi* is a
fixed point of the coarse-graining operator R_b (b-bin skeleton of perceived
match/reject rates) iff it is scale-invariant over a RANGE of b. Prediction:
VR and fire-viable form a PLATEAU over intermediate b (e.g. 3-8) and collapse
at both ends (b->1 too coarse: distinct causal states merged; b->inf exact:
no recurrence). A single sharp peak instead of a plateau falsifies M6.

Reuses the v1.3 skeleton addressing channel (address_key_fn). Domain fixed at
loop_w500_d00 (the best v1.3 condition) so the only variable is granularity b.

Usage: python poc/v14_granularity_scan.py <arm> <seed> <n_cycles>
Arms: b1 b2 b3 b4 b6 b8 b12 b16 exact
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexLoopDomain


def make_skel_key(nbins):
    def key(raw_obs):
        mr = raw_obs.get('match_result', {})
        if not isinstance(mr, dict):
            return 'na'
        m = mr.get('match_rate', 0.0)
        r = mr.get('reject_rate', 0.0)
        mb = min(int(m * nbins), nbins - 1)
        rb = min(int(r * nbins), nbins - 1)
        return f"m{mb}_r{rb}"
    return key


# arm -> (nbins or None for exact)
ARMS = {
    'b1': 1, 'b2': 2, 'b3': 3, 'b4': 4, 'b6': 6,
    'b8': 8, 'b12': 12, 'b16': 16, 'exact': None,
}


def main():
    arm, seed, n_cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nb = ARMS[arm]
    keyfn = None if nb is None else make_skel_key(nb)
    domain = RegexLoopDomain(seed=seed, task_duration=500, dissipation_rate=0.0)

    per_task = {}
    orig_act = domain.act

    def tracked_act(action_id):
        result = orig_act(action_id)
        acc = per_task.setdefault(result.metadata['task'], [0, 0])
        acc[0] += 1
        acc[1] += 1 if result.success else 0
        return result

    domain.act = tracked_act

    agent = TUFEAgent(domain, seed=seed, exact_addressing=True, address_key_fn=keyfn)
    t0 = time.time()
    report = agent.run(n_cycles=n_cycles, report_every=n_cycles)
    seconds = time.time() - t0

    ptf = {t: round(v / n, 3) for t, (n, v) in sorted(per_task.items())}
    out = {
        'arm': arm,
        'nbins': nb,
        'seed': seed,
        'viability_rate': report['viability_rate'],
        'confirmed_pivots': report['knowledge']['n_confirmed'],
        'replay_fires': agent.exact_fires,
        'replay_fire_viable_rate': round(agent.exact_fire_viable / max(agent.exact_fires, 1), 3),
        'fire_rate': round(agent.exact_fires / n_cycles, 4),
        'max_task_viable_fraction': max(ptf.values()) if ptf else 0.0,
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
