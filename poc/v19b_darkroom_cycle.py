"""v1.9b (REWRITE of retracted v1.9): ACTIVE INFORMATION ACQUISITION as a
complete work-cycle (刹那), with least-action decomposition. Pre-registered in
poc/preregister/v19b_darkroom_cycle.yaml under PROTOCOL_autonomous_run.md.

Correction of v1.9's two errors (see pivot-least-action-decomposition-grain):
  1. v1.9 coded "grope-to-switch" as a MAGIC ATOMIC action (PROBE => reveal h).
     Here it is decomposed into its minimal-action constituents: an actor with a
     position, a bounded target (switch_pos), a path/method (multi-step search),
     and a completion criterion (press at switch => light on => h acquired).
  2. v1.9 judged viability PER SINGLE STEP. Here viability is settled ONCE at
     COMPLETE-CYCLE completion (light on). No per-step reward, no discount, no
     value backprop => NOT RL. The cycle boundary is where the least action to
     the goal completes (a property of the task geometry) => NOT designer-drawn.

The information (switch_pos / h) is NOT in the passive observation stream; it can
only be ACQUIRED by the agent moving, feeling, and pressing => genuine active
info acquisition (spiral), not re-addressing of in-stream info (circle, v1.8).

Learning unit = the whole work-cycle: pivot = (start_bucket, path_signature,
cycle_viable). The METHOD (path) is composed by the agent's own uniform-random
exploration over the 4 primitives {left,right,feel,press}; NO "go to switch"
policy is supplied by the designer (G6/P-D).

Three arms (G2 negative controls):
  cycle_learn      : complete-cycle grain, learns (start -> path method), confirms
                     + replays + dissipative decay                       [main]
  explore_only     : same domain/grain but NEVER learns (pure random)    [floor]
  singlestep_grain : WRONG grain (per-step immediate viability, learns single
                     actions) — reproduces the retracted v1.9 pathology  [control]

Judgments (pre-registered, mutually exclusive):
  H-GRAIN   : cycle_learn VR >> explore_only floor AND > singlestep_grain, and the
              learned path truly navigates to switch_pos (P-D content match).
  H-NOGAIN  : cycle_learn ~= explore_only (grain fix insufficient).
  H-GRAINVAL: singlestep_grain ~= cycle_learn (grain was not the issue => the
              correction itself is falsified).
  H-LEAK    : learned path does not correspond to real switch / designer-supplied.

rho = 0 everywhere (no rescue). Usage: python poc/v19b_darkroom_cycle.py <arm> <seed> <n_cycles> [switch_mode]
"""
import json
import sys
import time

import numpy as np

L = 5                 # room cells 0..L-1
BUDGET = 12           # max sub-steps per work-cycle
N_ACT = 4             # 0=left, 1=right, 2=feel, 3=press
LEFT, RIGHT, FEEL, PRESS = 0, 1, 2, 3
SWITCH_FIXED = 2


# --------------------------------------------------------------------------
# Environment: dark room, switch at a bounded location, hidden bit h behind it
# --------------------------------------------------------------------------
class DarkRoom:
    def __init__(self, seed=42, switch_mode='fixed'):
        self.rng = np.random.default_rng(seed)
        self.switch_mode = switch_mode
        self.switch_pos = SWITCH_FIXED if switch_mode == 'fixed' else int(self.rng.integers(0, L))
        self.reset_cycle()

    def reset_cycle(self):
        self.pos = int(self.rng.integers(0, L))
        self.light_on = False
        self.h = int(self.rng.integers(0, 2))       # hidden until light on
        self.feel_at_switch = False                 # tactile flag, set by FEEL
        if self.switch_mode == 'moving':
            self.switch_pos = int(self.rng.integers(0, L))

    def observe(self):
        # passive stream: own position (proprioception), light, tactile flag.
        # switch_pos and h are NOT observable unless light_on / felt.
        return {
            'pos': self.pos,
            'light_on': self.light_on,
            'feel_at_switch': self.feel_at_switch,
            'h_visible': self.h if self.light_on else None,
        }

    def step(self, a):
        """Execute one sub-step. Returns (done, viable). viable is only True at
        cycle completion (press at switch => light on => h acquired)."""
        a = a % N_ACT
        if a == LEFT:
            self.pos = max(0, self.pos - 1)
            self.feel_at_switch = False
        elif a == RIGHT:
            self.pos = min(L - 1, self.pos + 1)
            self.feel_at_switch = False
        elif a == FEEL:
            self.feel_at_switch = (self.pos == self.switch_pos)
        elif a == PRESS:
            if self.pos == self.switch_pos:
                self.light_on = True                 # cycle completes
                return True, True
            self.feel_at_switch = False
        return self.light_on, self.light_on


# --------------------------------------------------------------------------
# Arm 1/2: complete-cycle-grain agent. Learning unit = the whole work-cycle.
# --------------------------------------------------------------------------
class CycleGrainAgent:
    def __init__(self, env, seed=42, learn=True, decay=0.002):
        self.env = env
        self.rng = np.random.default_rng(seed + 500)
        self.learn = learn
        self.decay = decay
        # methods[start_bucket] = {'path': tuple, 'n': int, 'v': int}
        self.methods = {}

    def _choose_sub(self, obs, replay_path):
        if replay_path:
            return replay_path.pop(0)
        # uniform-random exploration over the 4 primitives; NO supplied policy
        return int(self.rng.integers(0, N_ACT))

    def run_cycle(self):
        self.env.reset_cycle()
        start_bucket = self.env.observe()['pos']       # proprioceptive, observable
        replay_path = None
        if self.learn:
            m = self.methods.get(start_bucket)
            if m and m['n'] >= 1 and m['v'] >= 1 and self.rng.random() < 0.9:
                replay_path = list(m['path'])

        steps = []
        viable = False
        for _ in range(BUDGET):
            a = self._choose_sub(self.env.observe(), replay_path)
            steps.append(a)
            done, viable = self.env.step(a)
            if done:
                break

        # ---- viability settled ONCE, at complete-cycle grain ----
        if self.learn:
            sig = tuple(steps)
            m = self.methods.setdefault(start_bucket, {'path': None, 'n': 0, 'v': 0})
            m['n'] += 1
            if viable:
                m['v'] += 1
                # keep a viable method; prefer the shorter (least action, M12)
                if m['path'] is None or len(sig) < len(m['path']):
                    m['path'] = sig
            # dissipative decay: drop stale methods that keep failing
            if m['n'] >= 20 and m['v'] / m['n'] < 0.15:
                m['path'] = None
                m['n'] = m['v'] = 0
        return viable, steps

    def learned_paths(self):
        return {k: v['path'] for k, v in self.methods.items() if v['path']}


# --------------------------------------------------------------------------
# Arm 3: WRONG-grain control — viability judged per single sub-step, learns
# single (state -> action) immediate viability (reproduces retracted v1.9).
# --------------------------------------------------------------------------
class SingleStepGrainAgent:
    def __init__(self, env, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed + 500)
        self.mem = {}   # (pos, feel) -> {action: (n, v)} with PER-STEP viability

    def run_cycle(self):
        self.env.reset_cycle()
        viable = False
        for _ in range(BUDGET):
            obs = self.env.observe()
            key = (obs['pos'], int(obs['feel_at_switch']))
            rec = self.mem.setdefault(key, {})
            cand = [(v / n, a) for a, (n, v) in rec.items() if n >= 3 and v >= 0.5 * n]
            if cand and self.rng.random() < 0.8:
                a = int(max(cand)[1])
            elif self.rng.random() < 0.15:
                a = int(self.rng.integers(0, N_ACT))
            else:
                a = int(self.rng.integers(0, N_ACT))
            done, step_viable = self.env.step(a)
            # WRONG grain: credit/blame this single action by immediate viability
            n, v = rec.get(a, (0, 0))
            rec[a] = (n + 1, v + (1 if step_viable else 0))
            if done:
                viable = True
                break
        return viable, []


ARMS = {
    'cycle_learn':      dict(cls='cycle', learn=True),
    'explore_only':     dict(cls='cycle', learn=False),
    'singlestep_grain': dict(cls='single', learn=False),
}


def verify_path(env_factory, path, start_pos):
    """P-D content check: does the learned path actually navigate start->switch
    and press? Replay it on a fresh env with the same geometry."""
    env = env_factory()
    env.reset_cycle()
    env.pos = start_pos
    env.light_on = False
    for a in path:
        done, viable = env.step(a)
        if done:
            return viable
    return False


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    switch_mode = sys.argv[4] if len(sys.argv) > 4 else 'fixed'
    cfg = ARMS[arm]

    env = DarkRoom(seed=seed, switch_mode=switch_mode)
    if cfg['cls'] == 'cycle':
        agent = CycleGrainAgent(env, seed=seed, learn=cfg['learn'])
    else:
        agent = SingleStepGrainAgent(env, seed=seed)

    t0 = time.time()
    viable_cycles = 0
    second_half_viable = 0
    path_lengths = []
    for c in range(n_cycles):
        viable, steps = agent.run_cycle()
        viable_cycles += 1 if viable else 0
        if c >= n_cycles // 2:
            second_half_viable += 1 if viable else 0
        if viable:
            path_lengths.append(len(steps))
    seconds = time.time() - t0

    vr = viable_cycles / max(n_cycles, 1)
    vr_2nd = second_half_viable / max(n_cycles - n_cycles // 2, 1)

    # P-D content verification for cycle_learn
    pd_frac = None
    n_methods = 0
    if cfg['cls'] == 'cycle' and cfg['learn']:
        paths = agent.learned_paths()
        n_methods = len(paths)
        checked = 0
        ok = 0
        for start_pos, path in paths.items():
            if path is None:
                continue
            checked += 1
            if verify_path(lambda: DarkRoom(seed=seed, switch_mode=switch_mode), path, start_pos):
                ok += 1
        pd_frac = round(ok / checked, 3) if checked else None

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles, 'switch_mode': switch_mode,
        'VR_cycle': round(vr, 4),
        'VR_second_half': round(vr_2nd, 4),
        'mean_success_path_len': round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else None,
        'n_learned_methods': n_methods,
        'PD_path_verifies_frac': pd_frac,     # 1.0 => learned path truly reaches switch (env content)
        'switch_pos': env.switch_pos if switch_mode == 'fixed' else None,
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
