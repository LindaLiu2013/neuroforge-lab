"""v1.10: MOVING-TARGET conditional search policy via Fermat/M12 least-action
distillation + muscle-memory consolidation. Pre-registered in
poc/preregister/v110_moving_target_policy.yaml under PROTOCOL_autonomous_run.md.

Decision (C) locked by Jayson: the search is over WORK-CYCLES (random exploration
succeeds ~1/3 per cycle in L=5/BUDGET=12 = abundant raw material), NOT over the
~1e6 policy space, and NOT via a composition operator that would risk encoding the
sweep. Mechanism:
  1. one successful experience  : a viable cycle's executed action sequence is
                                  captured as a candidate METHOD (a path with phase
                                  memory; a memoryless (pos,feel) map cannot express
                                  a search because feel=False is ambiguous between
                                  "untested" and "tested-negative").
  2. repeated success -> muscle : the method is replayed; WHOLE-method binary cycle
     memory                       viability accumulates (n,v); fire criterion
                                  (n>=3 & v>=0.5n) consolidates it into replay.
  3. Fermat / M12 least action  : AMONG fire-eligible methods (those viable ACROSS
                                  varying switch positions), keep the SHORTEST. A
                                  replayed method that the random fallback had to
                                  EXTEND to succeed yields a longer captured path,
                                  so methods GROW toward coverage; M12 then prunes to
                                  the least-action one that covers => a systematic
                                  press/move sweep EMERGES, it is not encoded.

RED-LINE GUARDS (locked pre-run, see prereg):
  * NON-RL: the confirmed unit is the WHOLE method (path), by BINARY cycle viability,
    at complete-cycle grain (identical to v1.9b). NO per-step reward / discount /
    value backprop / TD. A method's v increments only when ITS OWN replayed action
    achieved the press (whole-method attribution), never by crediting individual
    state->action entries with a delayed outcome.
  * NON-HUMAN-DOOR: method content comes ONLY from the agent's own executed
    trajectories. No switch_pos / sweep / coverage / preset-direction token appears
    anywhere below (code-audit G6). M12 only selects "shortest viable", target-agnostic.
  * P-D: the consolidated method must re-verify on FRESH random switch positions.

Arms (G2 negative controls + human-door ceiling):
  policy_learn : (C) mechanism                                        [main]
  fixed_path   : v1.9b cycle_learn (start->fixed path, unconditional)  [neg control]
  explore_only : same domain/grain, NEVER learns (pure random)         [floor + raw-material rate]
  oracle_sweep : designer-SUPPLIED press/move coverage sweep           [human-door ceiling, NON-endogenous]

rho = 0 everywhere. Usage: python poc/v110_moving_target_policy.py <arm> <seed> <n_cycles> [switch_mode]
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

FIRE_N = 3            # muscle-memory threshold: min trials before a method can fire
FIRE_VRATIO = 0.5     # muscle-memory threshold: min cross-position viability ratio to fire
DECAY_N = 20          # dissipative decay: min trials before judging a method stale
DECAY_VRATIO = 0.5    # principled = FIRE_VRATIO: a method that can't reach the muscle-memory
                      # bar across many switch positions is not a real method -> fade it
REPLAY_P = 0.9        # replay a fire-eligible method with this prob (else explore)
MAX_METHODS = 300     # memory cap; drop least-promising when exceeded


# --------------------------------------------------------------------------
# Arm 1 (main): PolicyDistillAgent — decision (C) Fermat/M12 distillation.
# Learning unit = a WHOLE method (action path), confirmed at cycle grain.
# --------------------------------------------------------------------------
class PolicyDistillAgent:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed + 700)
        # each method: {'path': tuple(actions), 'n': int, 'v': int}
        self.methods = []

    # ---- selection: fire-eligible (viable across positions) THEN shortest (M12) ----
    def _best(self):
        elig = [m for m in self.methods
                if m['n'] >= FIRE_N and m['v'] >= FIRE_VRATIO * m['n']]
        if elig:
            return min(elig, key=lambda m: len(m['path']))     # muscle memory + Fermat/M12
        cands = [m for m in self.methods if m['n'] < FIRE_N]    # not yet consolidated:
        if cands:
            return min(cands, key=lambda m: len(m['path']))     # test least-action first
        return None

    def run_cycle(self):
        self.env.reset_cycle()                       # moving => fresh random switch_pos
        best = self._best()
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

        # ---- viability settled ONCE, at complete-cycle grain; WHOLE-method credit ----
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
            # keep the most promising: highest viability ratio, then least action (M12)
            self.methods.sort(key=lambda m: (m['v'] / max(m['n'], 1), -len(m['path'])))
            self.methods = self.methods[-MAX_METHODS:]

    def _decay(self):
        self.methods = [m for m in self.methods
                        if not (m['n'] >= DECAY_N and m['v'] < DECAY_VRATIO * m['n'])]

    def consolidated(self):
        elig = [m for m in self.methods
                if m['n'] >= FIRE_N and m['v'] >= FIRE_VRATIO * m['n']]
        return min(elig, key=lambda m: len(m['path'])) if elig else None


# --------------------------------------------------------------------------
# Arm 4 (human-door ceiling, NON-endogenous): designer-SUPPLIED coverage sweep.
# Press at each unvisited cell, sweep to one wall then the other. External door
# (OC-12): used ONLY to calibrate the value of the missing capability. NEVER
# counted toward "the system learned it". One env.step per iteration (budget-fair).
# --------------------------------------------------------------------------
class OracleSweepAgent:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed + 900)
        self.direction = RIGHT

    def run_cycle(self):
        self.env.reset_cycle()
        visited = set()
        self.direction = RIGHT if self.env.pos <= L // 2 else LEFT
        viable = False
        for _ in range(BUDGET):
            p = self.env.observe()['pos']
            if p not in visited:
                visited.add(p)
                done, viable = self.env.step(PRESS)     # SUPPLIED: press each new cell
            else:
                if p <= 0:
                    self.direction = RIGHT
                elif p >= L - 1:
                    self.direction = LEFT
                done, viable = self.env.step(self.direction)  # SUPPLIED: sweep + bounce
            if done:
                break
        return viable, []


# --------------------------------------------------------------------------
# P-D content verification for the moving target: does the consolidated method
# GENERALIZE to FRESH random switch positions? (the key P-D for a moving target)
# --------------------------------------------------------------------------
def verify_method_generalizes(seed, method, n_trials=200):
    if method is None:
        return None
    rng = np.random.default_rng(seed + 1234)
    env = DarkRoom(seed=seed + 99, switch_mode='moving')
    ok = 0
    for _ in range(n_trials):
        env.reset_cycle()                            # fresh random switch_pos each trial
        replay = list(method['path'])
        viable = False
        for _ in range(BUDGET):
            a = replay.pop(0) if replay else int(rng.integers(0, N_ACT))  # random fallback
            done, viable = env.step(a)
            if done:
                break
        ok += 1 if viable else 0
    return round(ok / n_trials, 3)


ARMS = {
    'policy_learn':  dict(cls='policy'),
    'fixed_path':    dict(cls='cycle', learn=True),    # v1.9b level-0 negative control
    'explore_only':  dict(cls='cycle', learn=False),   # random floor + raw-material rate
    'oracle_sweep':  dict(cls='oracle'),               # human-door ceiling
}


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    switch_mode = sys.argv[4] if len(sys.argv) > 4 else 'moving'
    cfg = ARMS[arm]

    env = DarkRoom(seed=seed, switch_mode=switch_mode)
    if cfg['cls'] == 'policy':
        agent = PolicyDistillAgent(env, seed=seed)
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
    if cfg['cls'] == 'policy':
        best = agent.consolidated()
        n_methods = len(agent.methods)
        if best is not None:
            pd_gen = verify_method_generalizes(seed, best)
            best_len = len(best['path'])

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles, 'switch_mode': switch_mode,
        'VR_cycle': round(vr, 4),
        'VR_second_half': round(vr_2nd, 4),
        'mean_success_path_len': round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else None,
        'n_methods': n_methods,
        'PD_generalizes_frac': pd_gen,        # P-D on FRESH switch positions (moving-target key check)
        'best_method_len': best_len,          # M12: length of the consolidated least-action method
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
