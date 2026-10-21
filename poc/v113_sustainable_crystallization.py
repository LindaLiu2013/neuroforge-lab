"""
v1.13 — chemostat: sustainable endogenous crystallization as a NON-TELEOLOGICAL
dissipative structure. See poc/preregister/v113_sustainable_crystallization.yaml
(COMMITTED BEFORE RESULTS, e146e34).

The agent is an artificial autocatalytic chemistry. It has NO goal, NO target, NO
oracle, and does NOT know its future. viability = self-maintenance (internal
free-energy budget stays > 0). capability crystallizes as a BYPRODUCT of keeping
itself coupled to a drifting environmental flux. Learning signal = feasible-space
dynamics, not reward [FS-C2]. Measured by OC-9 excess causal validity, never win-rate.

PHYSICAL CONSTANTS below were set ONCE by reasoning about the energy budget BEFORE
running (survival threshold coupling = (DECAY+ACT_COST)/GAIN = 0.433; a random op
couples at ~0.35 < threshold, so self-maintenance REQUIRES adaptation). They are NOT
tuned to results (K-NO-TUNE-TO-WIN / S4 / G5).

DESIGN-TIME CORRECTION (noted BEFORE running, anti-HARKing): the prereg's P-FROZEN
sub-clause predicted excess ~= background at drift=0 by analogy to the v1.5 "learning
illusion". That analogy is imperfect here: at drift=0 there is NO free environmental
rescue, so a frozen agent holds a REAL coupler (excess stays positive). The genuine
frozen signature in this mechanism is minting_rate -> 0 with static feasible volume
(capability crystallized once, zero renewal), NOT excess->background. P-FROZEN is
therefore evaluated primarily on the minting->0 clause; the excess value at drift=0 is
reported honestly whatever it is.
"""
import json
import random
import math

# ----------------------------- physical constants (frozen pre-run) -------------
N_CH = 6
DECAY = 1.0            # budget dissipated per step (P1.2)
GAIN = 3.0             # budget gained per successful coupling
ACT_COST = 0.3         # cost of firing any op
B0 = 25.0
BUDGET_MAX = 60.0
SENSE_NOISE = 0.05     # per-bit lossy sensing (P4), constant across arms
HORIZON = 4000
WINDOW = 100

CONC_INIT = 0.6
CONC_UP = 1.0
CONC_DOWN = 0.6
CONC_DECAY = 0.997     # global dissipation of every op per step (P1.2 forgetting)
CONC_MIN = 0.3
MAX_OPS = 40           # repertoire bandwidth cap (P4/P5)

N_BASE_OPS = 6
VAR_P = 0.20           # variation / minting probability per step (randomness door, OC-12)
MIN_N = 12
VIABLE_RATIO = 0.5

DRIFTS = [0.0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.40]
SEEDS = [42, 43, 44, 45]


class Op:
    __slots__ = ("mask", "conc", "n", "v", "novel", "consolidated")

    def __init__(self, mask, conc, novel=False):
        self.mask = mask
        self.conc = conc
        self.n = 0
        self.v = 0
        self.novel = novel
        self.consolidated = False


def random_base_mask(rng):
    w = rng.choice([1, 1, 2])          # small masks: reactions needing 1-2 channels
    bits = rng.sample(range(N_CH), w)
    m = 0
    for b in bits:
        m |= (1 << b)
    return m


ALL_PATTERNS = list(range(1, 1 << N_CH))


def couples(mask, p):
    return mask != 0 and (mask & p) == mask


def is_novel(nm, parent_masks):
    """Prereg-spec novelty: nm couples to SOME pattern that no parent could couple to."""
    if nm == 0:
        return False
    for p in ALL_PATTERNS:
        if couples(nm, p) and not any(couples(pm, p) for pm in parent_masks):
            return True
    return False


def _channels_covered(ops):
    """Saturation diagnostic: #distinct channels c for which a live single-channel op {c}
    exists. When this hits N_CH the repertoire can couple to EVERY non-empty pattern
    (feasible space saturated) -> nothing genuinely new left to crystallize."""
    return len({o.mask.bit_length() - 1 for o in ops if bin(o.mask).count("1") == 1})


def step_env(pattern, drift, rng):
    """Random-walk the flux pattern; keep >=1 channel active (vent always emits)."""
    if rng.random() < drift:
        while True:
            b = rng.randrange(N_CH)
            nxt = pattern ^ (1 << b)
            if nxt != 0:
                return nxt
    return pattern


def sense(pattern, rng):
    s = pattern
    for b in range(N_CH):
        if rng.random() < SENSE_NOISE:
            s ^= (1 << b)
    return s


def run_chemostat(drift, seed, adaptive=True):
    rng = random.Random(seed * 1000 + int(drift * 10000))
    pattern = 0
    while pattern == 0:
        pattern = rng.randrange(1, 1 << N_CH)

    ops = [Op(random_base_mask(rng), CONC_INIT) for _ in range(N_BASE_OPS)]
    # de-dup masks
    seen = set()
    uniq = []
    for o in ops:
        if o.mask not in seen:
            seen.add(o.mask)
            uniq.append(o)
    ops = uniq

    budget = B0
    alive = True
    novel_consol_events = []          # step index of each novel-consolidation
    windows = []                      # per-window log
    fires = succ = 0
    w_fires = w_succ = 0
    w_cover = 0

    for t in range(HORIZON):
        pattern = step_env(pattern, drift, rng)
        sensed = sense(pattern, rng)

        # --- feasibility of the repertoire in the CURRENT environment ---
        covered = any((o.mask & pattern) == o.mask for o in ops)
        if covered:
            w_cover += 1

        # --- fire: mass-action among triggered ops (no objective function) ---
        acted = False
        if adaptive:
            triggered = [o for o in ops if (o.mask & sensed)]
            if triggered:
                tot = sum(o.conc for o in triggered)
                x = rng.random() * tot
                acc = 0.0
                chosen = triggered[-1]
                for o in triggered:
                    acc += o.conc
                    if acc >= x:
                        chosen = o
                        break
                acted = True
        else:
            # frozen_random control: fixed repertoire, uniform firing, no adaptation
            triggered = [o for o in ops if (o.mask & sensed)]
            if triggered:
                chosen = rng.choice(triggered)
                acted = True

        if acted:
            budget -= ACT_COST
            fires += 1
            w_fires += 1
            ok = (chosen.mask & pattern) == chosen.mask
            if ok:
                budget += GAIN
                succ += 1
                w_succ += 1
                if adaptive:
                    chosen.conc += CONC_UP
                    chosen.v += 1
                    chosen.n += 1
            else:
                if adaptive:
                    chosen.conc = max(0.0, chosen.conc - CONC_DOWN)
                    chosen.n += 1

        budget -= DECAY
        if budget > BUDGET_MAX:
            budget = BUDGET_MAX

        if adaptive:
            # --- dissipation / forgetting of every op (P1.2) ---
            for o in ops:
                o.conc *= CONC_DECAY
            ops = [o for o in ops if o.conc >= CONC_MIN]
            if not ops:                       # repertoire fully decayed -> seed one random
                ops = [Op(random_base_mask(rng), CONC_INIT)]

            # --- consolidation check ---
            for o in ops:
                if (not o.consolidated) and o.n >= MIN_N and o.v / o.n >= VIABLE_RATIO:
                    o.consolidated = True
                    if o.novel:
                        novel_consol_events.append(t)

            # --- variation: mint candidate ops (randomness door, OC-12) ---
            if rng.random() < VAR_P and len(ops) < MAX_OPS:
                union = 0
                for o in ops:
                    union |= o.mask
                if rng.random() < 0.6:
                    parent = rng.choice(ops)
                    b = rng.randrange(N_CH)
                    nm = parent.mask ^ (1 << b)
                    if nm == 0:
                        nm = 1 << rng.randrange(N_CH)
                    parents = [parent.mask]
                else:
                    if len(ops) >= 2:
                        a, c = rng.sample(ops, 2)
                        nm = a.mask ^ c.mask
                        if nm == 0 or bin(nm).count("1") > 3:
                            nm = a.mask ^ (1 << rng.randrange(N_CH))
                            if nm == 0:
                                nm = 1 << rng.randrange(N_CH)
                        parents = [a.mask, c.mask]
                    else:
                        nm = random_base_mask(rng)
                        parents = []
                novel = is_novel(nm, parents)   # prereg-spec: couples to a parent-unreachable pattern
                ops.append(Op(nm, CONC_INIT, novel=novel))

            if len(ops) > MAX_OPS:             # bandwidth cap -> drop weakest
                ops.sort(key=lambda o: o.conc)
                ops = ops[-MAX_OPS:]

        # --- viability = self-maintenance ---
        if budget <= 0:
            alive = False
            budget = 0.0
            # record the death window then stop
            windows.append({
                "t": t, "budget": budget, "V": len(ops),
                "coverage": w_cover / WINDOW, "coupling": (w_succ / w_fires) if w_fires else 0.0,
                "novel_consol_cum": len(novel_consol_events), "alive": False,
                "channels_covered": _channels_covered(ops),
            })
            break

        # --- window logging ---
        if (t + 1) % WINDOW == 0:
            windows.append({
                "t": t + 1, "budget": budget, "V": len(ops),
                "coverage": w_cover / WINDOW,
                "coupling": (w_succ / w_fires) if w_fires else 0.0,
                "novel_consol_cum": len(novel_consol_events), "alive": True,
                "channels_covered": _channels_covered(ops),
            })
            w_fires = w_succ = w_cover = 0

    coupling_rate = (succ / fires) if fires else 0.0
    return {
        "drift": drift, "seed": seed, "adaptive": adaptive,
        "alive": alive, "survival_steps": (windows[-1]["t"] if windows else 0),
        "coupling_rate": coupling_rate,
        "novel_consol_total": len(novel_consol_events),
        "novel_consol_events": novel_consol_events,
        "final_V": len(ops),
        "windows": windows,
    }


def main():
    out = []
    for drift in DRIFTS:
        for seed in SEEDS:
            ad = run_chemostat(drift, seed, adaptive=True)
            fz = run_chemostat(drift, seed, adaptive=False)
            ad["background_rho"] = fz["coupling_rate"]
            ad["excess_causal_validity"] = ad["coupling_rate"] - fz["coupling_rate"]
            ad["frozen_alive"] = fz["alive"]
            out.append(ad)
            print(f"drift={drift:<5} seed={seed} alive={ad['alive']} "
                  f"surv={ad['survival_steps']:<5} V={ad['final_V']:<3} "
                  f"novelcons={ad['novel_consol_total']:<3} "
                  f"coupling={ad['coupling_rate']:.3f} rho={fz['coupling_rate']:.3f} "
                  f"excess={ad['excess_causal_validity']:+.3f} frozen_alive={fz['alive']}")
    with open("v113_results.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} runs to v113_results.jsonl")


if __name__ == "__main__":
    main()
