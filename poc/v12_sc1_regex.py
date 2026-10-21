"""v1.2 SC-1 falsifiable test: state-sufficient pivot addressing.

SC-1 (registered in v1.1): pivot replay must address a state representation
sufficient to reconstruct the causal context, otherwise confirmed knowledge
cannot be re-executed. This runner adds an exact-addressing causal memory
channel (full raw-state key) alongside the untouched bucket channel.

Pre-registered predictions (before any run):
  P1  control actions + exact addressing: VR >= 0.15 (control baseline
      0.0196) with per-task lock-in >= 0.5 on solvable tasks
      (starts with digit / contains number)
  P2  loop actions + exact addressing: VR >= 0.3 (6 of 8 tasks solvable)
  P3  bucket-channel confirmed pivots stay at control level (the exact
      channel must not leak into consensus metrics)
Falsification: exact addressing fails to lock in => SC-1 wrong; use
exact_fires / exact_fire_viable_rate to localize the true blocker.

Usage: python poc/v12_sc1_regex.py <control|control_exact|loop_exact> <seed> <n_cycles>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexDomain, RegexLoopDomain

ARMS = {
    'control': (RegexDomain, False),
    'control_exact': (RegexDomain, True),
    'loop_exact': (RegexLoopDomain, True),
}


def main():
    arm, seed, n_cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cls, exact = ARMS[arm]
    domain = cls(seed=seed)

    per_task = {}
    orig_act = domain.act

    def tracked_act(action_id):
        result = orig_act(action_id)
        acc = per_task.setdefault(result.metadata['task'], [0, 0])
        acc[0] += 1
        acc[1] += 1 if result.success else 0
        return result

    domain.act = tracked_act

    agent = TUFEAgent(domain, seed=seed, exact_addressing=exact)
    t0 = time.time()
    report = agent.run(n_cycles=n_cycles, report_every=n_cycles)
    seconds = time.time() - t0

    out = {
        'arm': arm,
        'seed': seed,
        'n_cycles': n_cycles,
        'viability_rate': report['viability_rate'],
        'confirmed_pivots': report['knowledge']['n_confirmed'],
        'exact_fires': agent.exact_fires,
        'exact_fire_viable_rate': round(agent.exact_fire_viable / max(agent.exact_fires, 1), 3),
        'per_task_viable_fraction': {
            t: round(v / n, 3) for t, (n, v) in sorted(per_task.items())
        },
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
