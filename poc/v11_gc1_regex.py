"""v1.1 GC-1 falsifiable test: open operation space in the Regex domain.

Pre-registered predictions (written before any treatment run):
  H1  VR(RegexOpen) >> VR(Regex) control (archived control: VR=0.018, 0 pivots)
  H2  confirmed pivots(RegexOpen) > 0
  H3  per-task viability rises exactly on tasks whose required constructor
      exists only in the open arm (has uppercase / has special char /
      two words / email-like); tasks solvable in both arms (starts with
      digit / contains number) stay similar. H3 isolates the mechanism to
      action-space richness rather than exploration noise.
Falsification: VR(open) ~ VR(control) and pivots == 0 => operation-space
richness is not the binding constraint behind GC-1.

Only the action set differs between arms; observers, viability rule,
dissipation, tasks and meta-rules are inherited unchanged.

Usage: python poc/v11_gc1_regex.py <control|open|loop> <seed> <n_cycles> [task_duration]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexDomain, RegexLoopDomain, RegexOpenDomain

ARMS = {'control': RegexDomain, 'open': RegexOpenDomain, 'loop': RegexLoopDomain}


def main():
    arm, seed, n_cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    domain = ARMS[arm](seed=seed)
    if len(sys.argv) > 4:
        domain._task_duration = int(sys.argv[4])

    per_task = {}
    orig_act = domain.act

    def tracked_act(action_id):
        result = orig_act(action_id)
        acc = per_task.setdefault(result.metadata['task'], [0, 0])
        acc[0] += 1
        acc[1] += 1 if result.success else 0
        return result

    domain.act = tracked_act

    agent = TUFEAgent(domain, seed=seed)
    t0 = time.time()
    report = agent.run(n_cycles=n_cycles, report_every=n_cycles)
    seconds = time.time() - t0

    out = {
        'arm': arm,
        'domain': domain.name(),
        'seed': seed,
        'n_cycles': n_cycles,
        'action_space': domain.action_space_size(),
        'viability_rate': report['viability_rate'],
        'confirmed_pivots': report['knowledge']['n_confirmed'],
        'phase_transitions': report['topology']['n_transitions'],
        'feasible_volume': report['feasible_space']['current_volume'],
        'feasible_trend': report['feasible_space']['trend'],
        'per_task_viable_fraction': {
            t: round(v / n, 3) for t, (n, v) in sorted(per_task.items())
        },
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
