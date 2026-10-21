"""v1.12: ENDOGENOUS CRYSTALLIZATION — can the system grow a CLOSED-LOOP reactive
coverage policy (approach oracle 1.0) with NO coverage answer supplied? Attacks the
SC-3 GENERATION gap that v1.11 localized. Pre-registered in
poc/preregister/v112_endogenous_crystallization.yaml under PROTOCOL_autonomous_run.md.

Jayson's directive: light refracting in water is PASSIVE; endogenous crystallization
(emergence / the system summarizing its OWN successful methods) is what this original
agent design must achieve.

Why v1.11 capped at ~0.42 (the real wall): oracle_sweep wins not by being smarter but
by being CLOSED-LOOP — it reacts to observations (feel a cell -> if switch, press; if
not, remember it ruled-out and move to an un-ruled cell). Our agent's method was an
OPEN-LOOP fixed action sequence, which structurally cannot cover a MOVING target. So
crystallization must RE-REPRESENT the method from a fixed path to a reactive policy.

Human-door judgment (declared, pre-registered as kill condition H-SUPPLIED): systematic
search needs WITHIN-CYCLE MEMORY of ruled-out cells. That memory is the agent retaining
its OWN past FEEL observations = re-addressing information already in its stream, which
v1.8 (P-A) confirmed legitimate (zero new info, OC-12 L1). The coverage FORM must still
emerge from environment selection; no coverage/sweep/direction/visited-as-target token
is supplied anywhere below.

Mechanism (all endogenous):
  * representation : genome[320], a closed-loop policy over state index
                     idx = pos*64 + feel*32 + ruled, ruled = 5-bit mask of cells the
                     agent FELT-and-found-negative THIS cycle (its own observation memory).
  * variation      : random mutation of some states' actions (OC-12 randomness door).
  * self-summary   : tally (state->action) over ALL winning cycles; on reproduction,
                     write tally-argmax back into offspring (Lamarckian) = "summarize
                     what I did when I succeeded".
  * selection      : WHOLE-policy binary cycle viability over a window (dissipative:
                     survivors maintained, failures fade). NON-RL: no per-step reward,
                     no discount, no value backprop, no TD, no gradient. Viability is
                     settled ONCE per complete cycle (刹那 grain, identical to v1.9b/v1.11).

Honest boundary (declared pre-run): the frozen meta-rules (mutation + self-summary +
whole-policy viability selection) are the designer-frozen FORM (π axis, same status as
v1.8's compression operator). What is ENDOGENOUS is WHICH reactive policy survives
(CONTENT, environment-verified) — NOT the invention of the search procedure itself.

Arms (G2 negative control + human-door ceiling; openloop 0.422 / explore 0.334 cited
from v1.11 same moving domain):
  reactive_crystallize : reactive policy + variation + self-summary + viability selection [main]
  reactive_no_learn    : same reactive representation, FROZEN random genome (no learn)   [rep-only control]
  oracle_sweep         : designer-SUPPLIED coverage sweep                                 [human-door ceiling]

rho = 0 everywhere. Usage: python poc/v112_endogenous_crystallization.py <arm> <seed> <n_cycles> [switch_mode]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v19b_darkroom_cycle import (          # reuse the decomposed dark-room env
    DarkRoom, L, BUDGET, N_ACT, LEFT, RIGHT, FEEL, PRESS,
)
from v110_moving_target_policy import OracleSweepAgent   # human-door ceiling

# --- frozen pre-run (see prereg; NOT tuned after results, G5/S4) ---
N_STATES = L * 2 * (1 << L)   # pos(5) * feel(2) * ruled(32) = 320
K = 8               # population size
W = 25              # cycles used to evaluate one genome's whole-policy viability
MUT_STATES = 6      # states randomly re-flipped per offspring
MIN_SUPPORT = 3     # min tally support before a state's summary can be written back
LAMARK_P = 0.5      # prob of writing tally-argmax back per supported state (self-summary)
ELITE = 2           # survivors carried unchanged to the next generation


def _idx(pos, feel, ruled):
    return pos * 64 + feel * 32 + ruled


def run_genome_cycle(env, genome, rng, tally=None):
    """Live ONE complete work-cycle under a closed-loop reactive genome.
    Viability is settled ONCE, at cycle completion (light on). If `tally` is given and
    the cycle is viable, the (state->action) pairs experienced are self-summarized into
    it (whole-cycle success only; NOT per-step causal credit)."""
    env.reset_cycle()                          # moving => fresh random switch_pos
    ruled = 0                                  # agent's OWN within-cycle observation memory
    experienced = [] if tally is not None else None
    viable = False
    for _ in range(BUDGET):
        obs = env.observe()
        idx = _idx(obs['pos'], 1 if obs['feel_at_switch'] else 0, ruled)
        a = int(genome[idx]) % N_ACT
        if tally is not None:
            experienced.append((idx, a))
        done, viable = env.step(a)
        if a == FEEL and not env.feel_at_switch:
            ruled |= (1 << env.pos)            # retain: this cell felt-and-negative (v1.8 re-addressing)
        if done:
            break
    if viable and tally is not None:
        for idx, a in experienced:             # self-summarize the winning experience
            tally[idx, a] += 1
    return viable


def _reproduce(pop, fit, tally, rng):
    order = np.argsort(fit)[::-1]
    new_pop = [pop[i].copy() for i in order[:ELITE]]
    weights = fit + 1e-6
    while len(new_pop) < K:
        parent = pop[int(rng.choice(len(pop), p=weights / weights.sum()))]
        child = parent.copy()
        for s in rng.choice(N_STATES, size=MUT_STATES, replace=False):   # random variation
            child[s] = rng.integers(0, N_ACT)
        supported = np.where(tally.sum(axis=1) >= MIN_SUPPORT)[0]         # self-summary source
        for idx in supported:
            if rng.random() < LAMARK_P:
                child[idx] = int(np.argmax(tally[idx]))
        new_pop.append(child)
    return new_pop


def _reactivity(genome):
    """Diagnostics: does the crystallized policy actually USE feel / ruled (reactive)?"""
    uses_feel = False
    uses_ruled = False
    for pos in range(L):
        for ruled in range(1 << L):
            if len({int(genome[_idx(pos, f, ruled)]) for f in (0, 1)}) > 1:
                uses_feel = True
        for feel in (0, 1):
            if len({int(genome[_idx(pos, feel, r)]) for r in range(1 << L)}) > 1:
                uses_ruled = True
    return uses_feel, uses_ruled


def verify_reactive(seed, genome, n_trials=300):
    """P-D: does the crystallized policy generalize to FRESH random switch positions?"""
    rng = np.random.default_rng(seed + 4321)
    env = DarkRoom(seed=seed + 77, switch_mode='moving')
    ok = 0
    for _ in range(n_trials):
        ok += 1 if run_genome_cycle(env, genome, rng) else 0
    return round(ok / n_trials, 3)


def run_crystallize(env, seed, n_cycles):
    rng = np.random.default_rng(seed + 300)
    tally = np.zeros((N_STATES, N_ACT), dtype=np.int64)
    pop = [rng.integers(0, N_ACT, size=N_STATES).astype(np.int8) for _ in range(K)]
    outcomes = []
    best_genome = pop[0].copy()
    best_fit = 0.0
    gens = 0
    while len(outcomes) + K * W <= n_cycles:
        fit = np.zeros(K)
        for i, g in enumerate(pop):
            v = 0
            for _ in range(W):
                ok = run_genome_cycle(env, g, rng, tally)   # eval cycles are lived cycles
                outcomes.append(ok)
                v += ok
            fit[i] = v / W
        if fit.max() > best_fit:
            best_fit = float(fit.max())
            best_genome = pop[int(np.argmax(fit))].copy()
        pop = _reproduce(pop, fit, tally, rng)
        gens += 1
    # exploit the best crystallized genome on any leftover budget
    while len(outcomes) < n_cycles:
        outcomes.append(run_genome_cycle(env, best_genome, rng, tally))
    return outcomes, best_genome, best_fit, gens


def run_no_learn(env, seed, n_cycles):
    rng = np.random.default_rng(seed + 300)
    genome = rng.integers(0, N_ACT, size=N_STATES).astype(np.int8)   # frozen random reactive policy
    outcomes = [run_genome_cycle(env, genome, rng) for _ in range(n_cycles)]
    return outcomes, genome, 0.0, 0


def main():
    arm = sys.argv[1]
    seed = int(sys.argv[2])
    n_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    switch_mode = sys.argv[4] if len(sys.argv) > 4 else 'moving'

    env = DarkRoom(seed=seed, switch_mode=switch_mode)
    t0 = time.time()

    if arm == 'reactive_crystallize':
        outcomes, best, best_fit, gens = run_crystallize(env, seed, n_cycles)
    elif arm == 'reactive_no_learn':
        outcomes, best, best_fit, gens = run_no_learn(env, seed, n_cycles)
    elif arm == 'oracle_sweep':
        agent = OracleSweepAgent(env, seed=seed)
        outcomes = [agent.run_cycle()[0] for _ in range(n_cycles)]
        best, best_fit, gens = None, 1.0, 0
    else:
        raise SystemExit('unknown arm: ' + arm)
    seconds = time.time() - t0

    n = len(outcomes)
    vr = sum(outcomes) / max(n, 1)
    vr_2nd = sum(outcomes[n // 2:]) / max(n - n // 2, 1)

    pd_gen = None
    uses_feel = uses_ruled = None
    if best is not None:
        pd_gen = verify_reactive(seed, best)
        uses_feel, uses_ruled = _reactivity(best)

    out = {
        'arm': arm, 'seed': seed, 'n_cycles': n_cycles, 'switch_mode': switch_mode,
        'VR_cycle': round(vr, 4),
        'VR_second_half': round(vr_2nd, 4),
        'best_policy_viability': round(best_fit, 4),   # elite whole-policy viability (oracle=1.0)
        'PD_generalizes_frac': pd_gen,                 # P-D on FRESH switch positions
        'uses_feel': uses_feel,                        # reactive diagnostic (closed-loop?)
        'uses_ruled': uses_ruled,                      # uses own ruled-out memory?
        'n_generations': gens,
        'seconds': round(seconds, 1),
    }
    print('RESULT_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
