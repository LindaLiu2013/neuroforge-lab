"""v1.3 SC-2 falsifiable test: M3 timing-mismatch vs action-destructiveness.

Distilled model M3 (see memory distilled-math-models): preference lock-in
needs T* = ln(15)/[f*ln1.08 + (1-f)*(-ln0.97)] cycles, but the task window
W=200 resets the pattern first. With measured f~0.02 on Regex, T*~4700 >> W,
so lock-in is mathematically impossible. Competing explanation: dissipation
(p=0.05) destroys any viable pattern each cycle, capping viability at 0.95
regardless of window.

This runner separates the two blockers with a 2x2 factorial on the loop
domain (which has hold/trim so a viable pattern CAN be preserved):
    window  in {200, 500}   (T* headroom)
    dissip  in {0.05, 0.0}  (pattern destruction)
All arms use bucket-only addressing (exact_addressing=False) so we test the
native preference/pivot channel's ability to concentrate.

Pre-registered predictions (before any run):
  P3a  loop_w500_d00 (both blockers removed): overall VR >= 0.15 AND at
       least one task with per-task viable fraction >= 0.5 (lock-in).
  P3b  If P3a fails (VR stays ~0.02, no task >= 0.5) => M3 wrong, the real
       blocker is action destructiveness / task difficulty, NOT timing.
       Granularity schemes (A/B arms) would then be moot.
  P3c  Monotonicity: VR(w500_d00) >= VR(w500_d05) >= VR(w200_d05) and
       VR(w500_d00) >= VR(w200_d00). Window and dissipation each help.
  P3d  d00 arms must beat d05 arms at equal window (dissipation is a real
       destroyer); if not, dissipation is not the competing blocker.

Falsification logic:
  - Lock-in appears only in w500 arms  => timing (M3) is the blocker.
  - Lock-in appears in d00 arms regardless of window => destruction is the blocker.
  - Neither alone suffices, only w500_d00 => both blockers real (M3 + destruction).
  - w500_d00 still fails => SC-2 granularity hope dead for Regex; redesign actions.

Usage: python poc/v13_sc2_regex.py <arm> <seed> <n_cycles>
Arms: loop_w200_d05 | loop_w500_d05 | loop_w200_d00 | loop_w500_d00
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tufe import TUFEAgent
from tufe.examples import RegexLoopDomain


def skeleton_key(raw_obs):
    """M2 skeleton: coarse (match_rate, reject_rate) bins, pattern string dropped.

    Sufficient statistic for the MAINTENANCE policy: any viable state maps to
    a high-high bin, so 'hold when viable' generalizes across the many distinct
    viable pattern strings that exact addressing treats as unrelated keys.
    """
    mr = raw_obs.get('match_result', {})
    if not isinstance(mr, dict):
        return 'na'
    m = mr.get('match_rate', 0.0)
    r = mr.get('reject_rate', 0.0)
    mb = min(int(m * 4), 3)
    rb = min(int(r * 4), 3)
    return f"m{mb}_r{rb}"


# arm -> (task_duration, dissipation_rate, exact_addressing, key_fn)
ARMS = {
    # P3 factorial: bucket-only native channel (no replay), tests M3 timing vs destruction
    'loop_w200_d05': (200, 0.05, False, None),
    'loop_w500_d05': (500, 0.05, False, None),
    'loop_w200_d00': (200, 0.00, False, None),
    'loop_w500_d00': (500, 0.00, False, None),
    # A arm: skeleton addressing (M2), replay 'hold when viable'
    'loop_skel_w200_d00': (200, 0.00, True, skeleton_key),
    'loop_skel_w500_d00': (500, 0.00, True, skeleton_key),
    'loop_skel_w200_d05': (200, 0.05, True, skeleton_key),
}


def main():
    arm, seed, n_cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dur, diss, exact, keyfn = ARMS[arm]
    domain = RegexLoopDomain(seed=seed, task_duration=dur, dissipation_rate=diss)

    per_task = {}
    orig_act = domain.act

    def tracked_act(action_id):
        result = orig_act(action_id)
        acc = per_task.setdefault(result.metadata['task'], [0, 0])
        acc[0] += 1
        acc[1] += 1 if result.success else 0
        return result

    domain.act = tracked_act

    agent = TUFEAgent(domain, seed=seed, exact_addressing=exact, address_key_fn=keyfn)
    t0 = time.time()
    report = agent.run(n_cycles=n_cycles, report_every=n_cycles)
    seconds = time.time() - t0

    ptf = {t: round(v / n, 3) for t, (n, v) in sorted(per_task.items())}
    locked = sorted((t for t, f in ptf.items() if f >= 0.5), key=lambda t: -ptf[t])

    out = {
        'arm': arm,
        'seed': seed,
        'n_cycles': n_cycles,
        'task_duration': dur,
        'dissipation_rate': diss,
        'viability_rate': report['viability_rate'],
        'confirmed_pivots': report['knowledge']['n_confirmed'],
        'replay_fires': agent.exact_fires,
        'replay_fire_viable_rate': round(agent.exact_fire_viable / max(agent.exact_fires, 1), 3),
        'max_task_viable_fraction': max(ptf.values()) if ptf else 0.0,
        'n_locked_tasks': len(locked),
        'locked_tasks': locked,
        'per_task_viable_fraction': ptf,
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
