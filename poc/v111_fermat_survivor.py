"""v1.11: FERMAT/FEYNMAN CORRECTION of v1.10. Re-encode M12 "least action" as the
STATIONARY SURVIVOR of the ensemble path-sum, NOT the naive global argmin-length.
Pre-registered in poc/preregister/v111_fermat_survivor.yaml under
PROTOCOL_autonomous_run.md.

Jayson's correction (v1.10 was the naive "least distance" misreading of Fermat):
  * Fermat is least TIME (water refraction proves it is not least distance), more
    strictly STATIONARY time (delta = 0).
  * Feynman path integral: light effectively traverses ALL paths; out-of-step paths
    cancel (destructive), stationary-action paths reinforce (constructive). The
    OBSERVED path is the PHENOMENAL SURVIVOR of the sum over the ensemble, not an
    absolutely pre-selected shortest path.

Corrected M12 encoding (three pieces, each maps to the physics):
  1. WALK ALL PATHS (ensemble): keep sampling under-trialed methods into the sum, so
     every captured path contributes; do NOT only replay the current best.
  2. SURVIVAL AMPLITUDE (interference): a path's amplitude = its cross-switch-position
     viability v/n, SHRUNK toward the random base rate for small n. Small-sample lucky
     shortcuts cancel (pulled back to base); large-sample robust methods survive.
        survival_score(m) = (v + SHRINK_V) / (n + SHRINK_N)
  3. LEAST ACTION = STATIONARY REFINEMENT ONLY (tie-break): among methods of ~EQUAL
     survival amplitude (within SURV_EPS) prefer the shorter. NEVER a global argmin-len.
  A method must also have n >= MIN_SURVIVE_N ("walked enough") to be a survivor.

RED-LINE GUARDS (locked pre-run, see prereg): identical to v1.10.
  * NON-RL: whole-method (path) binary cycle viability at complete-cycle grain; no
    per-step reward / discount / value backprop / TD. v increments only when the
    method's OWN replayed action achieved the press.
  * NON-HUMAN-DOOR (G6 code audit): selection uses ONLY the endogenous statistics
    (n, v, len). No switch_pos / sweep / coverage / preset-direction token. The shrink
    base rate ~= the explore random floor (measurable), not an injected answer.
  * P-D: the survivor must re-verify on FRESH random switch positions.

Arms (G2 negative controls + human-door ceiling):
  survivor_policy : corrected M12 (stationary survivor)                    [main]
  fixed_path      : v1.9b cycle_learn (start->fixed path, unconditional)   [neg control]
  explore_only    : same domain/grain, NEVER learns (pure random)          [floor + raw-material rate]
  oracle_sweep    : designer-SUPPLIED press/move coverage sweep            [human-door ceiling, NON-endogenous]
(v1.10 policy_learn = the naive least-distance result, 0.44, is cited as the
 historical control, NOT re-run.)

rho = 0 everywhere. Usage: python poc/v111_fermat_survivor.py <arm> <seed> <n_cycles> [switch_mode]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v19b_darkroom_cycle import (          # reuse the decomposed dark-room env
    DarkRoom, CycleGrainAgent, L, BUDGET, N_ACT, LEFT, RIGHT, FEEL, PRESS,
)
from v110_moving_target_policy import (    # reuse oracle ceiling + P-D verifier
    OracleSweepAgent, verify_method_generalizes,
)

# --- frozen pre-run (see prereg; NOT tuned after results, G5/S4) ---
MIN_SURVIVE_N = 15      # a method must be "walked" this many times to be a survivor
SHRINK_N = 10           # small-sample shrinkage strength toward the base rate
SHRINK_V = 3            # ~= base_rate * SHRINK_N (random explore floor ~0.3)
SURV_EPS = 0.05         # survival amplitudes within this are "equal" -> least-action tie-break
SUM_OVER_PATHS_P = 0.3  # prob of sampling an UNDER-trialed path into the ensemble sum
REPLAY_P = 0.9          # replay the selected method with this prob (else pure random explore)
DECAY_N = 30            # dissipative decay: min trials before judging a method destructive
DECAY_VRATIO = 0.35     # persistently below the random base rate -> cancels out -> fade it
MAX_METHODS = 300       # memory cap


class SurvivorPolicyAgent:
    """Decision (C) with M12 corrected to the ensemble stationary survivor."""

    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed + 700)
        self.methods = []        # each: {'path': tuple(actions), 'n': int, 'v': int}

    # ---- survival amplitude: cross-position viability shrunk toward base rate ----
    @staticmethod
    def _amp(m):
        return (m['v'] + SHRINK_V) / (m['n'] + SHRINK_N)

    # ---- selection = walk-all-paths ensemble + stationary survivor + least-action tie-break ----
    def _pick(self):
        if not self.methods:
            return None
        # (1) WALK ALL PATHS: with prob SUM_OVER_PATHS_P sample an under-trialed path
        #     into the sum so it can accumulate trials (destructive/constructive needs
        #     every contribution). Chosen uniformly => no designer bias among paths.
        under = [m for m in self.methods if m['n'] < MIN_SURVIVE_N]
        if under and self.rng.random() < SUM_OVER_PATHS_P:
            return under[int(self.rng.integers(0, len(under)))]
        # (2) STATIONARY SURVIVOR: among walked-enough methods, highest survival amplitude.
        survivors = [m for m in self.methods if m['n'] >= MIN_SURVIVE_N]
        pool = survivors if survivors else self.methods
        best_amp = max(self._amp(m) for m in pool)
        # (3) LEAST ACTION AS TIE-BREAK ONLY: among ~equal-amplitude methods, shortest.
        tied = [m for m in pool if best_amp - self._amp(m) < SURV_EPS]
        return min(tied, key=lambda m: len(m['path']))

    def run_cycle(self):
        self.env.reset_cycle()                       # moving => fresh random switch_pos
        best = self._pick()
        replay = list(best['path']) if (best is not None and self.rng.random() < REPLAY_P) else None
        used_replay = replay is not None

        executed = []
        viable = False
        achieved_by_replay = False
        for _ in range(BUDGET):
            from_replay = bool(replay)
            a = replay.pop(0) if replay else int(self.rng.integers(0, N_ACT))  # NO supplied policy
            executed.append(a)
            done, viable = self.env.step(a)
            if done:
                achieved_by_replay = from_replay      # did the method's OWN action win it?
                break

        # ---- viability settled ONCE at complete-cycle grain; WHOLE-method credit (non-RL) ----
        if used_replay and best is not None:
            best['n'] += 1
            if viable and achieved_by_replay:
                best['v'] += 1                        # re-confirmed on a FRESH switch position
        if viable:
            path = tuple(executed)
            if not (used_replay and best is not None and path == best['path']):
                self._capture(path)                   # one successful experience (or an extension)

        self._decay()
        return viable, executed

    def _capture(self, path):
        for m in self.methods:
            if m['path'] == path:
                m['n'] += 1
                m['v'] += 1                           # this success re-confirms it (whole unit)
                return
        self.methods.append({'path': path, 'n': 1, 'v': 1})
        if len(self.methods) > MAX_METHODS:
            # drop the lowest-amplitude (most destructive) contributions first
            self.methods.sort(key=self._amp)
            self.methods = self.methods[-MAX_METHODS:]

    def _decay(self):
        self.methods = [m for m in self.methods
                        if not (m['n'] >= DECAY_N and m['v'] < DECAY_VRATIO * m['n'])]

    def consolidated(self):
        survivors = [m for m in self.methods if m['n'] >= MIN_SURVIVE_N]
        if not survivors:
            return None
        best_amp = max(self._amp(m) for m in survivors)
        tied = [m for m in survivors if best_amp - self._amp(m) < SURV_EPS]
        return min(tied, key=lambda m: len(m['path']))


ARMS = {
    'survivor_policy': dict(cls='survivor'),
    'fixed_path':      dict(cls='cycle', learn=True),    # v1.9b level-0 negative control
    'explore_only':    dict(cls='cycle', learn=False),   # random floor + raw-material rate
    'oracle_sweep':    dict(cls='oracle'),               # human-door ceiling
}


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    switch_mode = sys.argv[4] if len(sys.argv) > 4 else 'moving'
    cfg = ARMS[arm]

    env = DarkRoom(seed=seed, switch_mode=switch_mode)
    if cfg['cls'] == 'survivor':
        agent = SurvivorPolicyAgent(env, seed=seed)
    elif cfg['cls'] == 'oracle':
        agent = OracleSweepAgent(env, seed=seed)
    else:
        agent = CycleGrainAgent(env, seed=seed, learn=cfg['learn'])

    t0 = time.time()
    viable_cycles = 0
    second_half = 0
    path_lengths = []
    for c in range(n_cycles):
        viable, steps = agent.run_cycle()
        viable_cycles += 1 if viable else 0
        if c >= n_cycles // 2:
            second_half += 1 if viable else 0
        if viable and steps:
            path_lengths.append(len(steps))
    seconds = time.time() - t0

    vr = viable_cycles / max(n_cycles, 1)
    vr_2nd = second_half / max(n_cycles - n_cycles // 2, 1)

    pd_gen = None
    n_methods = 0
    best_len = None
    best_amp = None
    if cfg['cls'] == 'survivor':
        best = agent.consolidated()
        n_methods = len(agent.methods)
        if best is not None:
            pd_gen = verify_method_generalizes(seed, best)
            best_len = len(best['path'])
            best_amp = round(SurvivorPolicyAgent._amp(best), 3)

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles, 'switch_mode': switch_mode,
        'VR_cycle': round(vr, 4),
        'VR_second_half': round(vr_2nd, 4),
        'mean_success_path_len': round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else None,
        'n_methods': n_methods,
        'PD_generalizes_frac': pd_gen,        # P-D on FRESH switch positions (moving-target key check)
        'best_method_len': best_len,          # length of the stationary survivor (NOT forced shortest)
        'best_method_amp': best_amp,          # survival amplitude v/n shrunk (interference strength)
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
