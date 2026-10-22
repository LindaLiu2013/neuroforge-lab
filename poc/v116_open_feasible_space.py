"""
v1.16 — attack F1 (saturation ceiling): does a COMBINATORIALLY-OPEN feasible space turn
the v1.13 frozen saturation into SUSTAINED endogenous crystallization?
See poc/preregister/v116_open_feasible_space.yaml (COMMITTED BEFORE RESULTS, 1d5d2d1).

Non-teleological frame inherited verbatim from v1.13. The agent has NO goal, NO target,
NO oracle, and does NOT know its future. viability = self-maintenance (budget>0). capability
crystallizes as a BYPRODUCT of staying coupled to a drifting flux. Measured by OC-9 excess +
coverage + survival + minting continuity, NEVER win-rate.

MINIMAL SINGLE-VARIABLE CONTRAST: the ONLY swept structural variable is world size N_CH.
  SMALL (N_CH=8) : full coupler pair-space C(8,2)=28 FITS the op budget (40) -> SATURABLE -> F1.
  OPEN  (N_CH=64): pair-space C(64,2)=2016 >> 40 -> UNSATURABLE -> always something new to mint.
All v1.13 physical constants, the drift grid, and the seeds are IDENTICAL across both worlds.

Three structural choices are FORCED (design-time, applied identically to both worlds, NOT tuned):
  1. ops = weight-2 SPECIFIC couplers (subset success), because weight-1 ops are so general that
     ~N_CH of them saturate ANY world -> openness could not be expressed by world size alone.
  2. environment patterns = weight-3, random walk by swapping ONE channel (local drift).
  3. variation preserves weight-2 and is LOCAL (one-channel swap / two-parent recombination),
     matching the local drift so the OPEN agent can TRACK the walk but never FINISH (N=64 pairs).
Cold-start guarantee: exactly one initial coupler matches the INITIAL pattern (identical rule both
worlds), so OPEN can bootstrap (random coupling 0.0015 -> 99% die without it). It encodes NOTHING
about the drift or the future; it goes obsolete as soon as the pattern moves. [K-COLD-START-HONESTY]
Sensing is CONSTANT-DIFFICULTY (one error w.p. SENSE_P, not v1.13 per-bit) so openness is not
confounded with sensing difficulty.

Physical constants were set ONCE by v1.13's energy-budget reasoning and are NOT re-tuned here
(K-NO-TUNE-TO-WIN / S4 / G5): survival threshold coupling = (DECAY+ACT_COST)/GAIN = 0.4333; a
random weight-2 coupler couples at 0.1071 (SMALL) / 0.0015 (OPEN), both < threshold, so
self-maintenance REQUIRES adaptation in BOTH worlds (no free environmental rescue).
"""
import json
import random
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- physical constants (IDENTICAL to v1.13) -------------
DECAY = 1.0
GAIN = 3.0
ACT_COST = 0.3
B0 = 25.0
BUDGET_MAX = 60.0
HORIZON = 4000
WINDOW = 100

CONC_INIT = 0.6
CONC_UP = 1.0
CONC_DOWN = 0.6
CONC_DECAY = 0.997
CONC_MIN = 0.3
MAX_OPS = 40

N_BASE_OPS = 6
VAR_P = 0.20
MIN_N = 12
VIABLE_RATIO = 0.5

# ------------------- v1.16 structural choices (pre-registered, both worlds) --------
OP_WEIGHT = 2
PATTERN_WEIGHT = 3
SENSE_P = 0.30          # constant-difficulty: ONE sensing error w.p. SENSE_P (matches v1.13 expected error count 6*0.05=0.3)

WORLDS = {"SMALL": 8, "OPEN": 64}
DRIFTS = [0.0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.40]
SEEDS = [42, 43, 44, 45]

# ------------------- frozen judgment thresholds (from prereg) ----------------------
EXCESS_MARGIN = 0.05
SATURATION_CUT = 0.90
SUSTAIN_TAIL_RATIO = 0.5
FROZEN_TAIL_RATIO = 0.2
SURVIVAL_HIGH = 0.9
SURVIVAL_COLLAPSE = 0.5
SLOPE_EPS = 0.10


class Op:
    __slots__ = ("mask", "conc", "n", "v", "novel", "new_mask", "consolidated")

    def __init__(self, mask, conc, novel=False, new_mask=False):
        self.mask = mask
        self.conc = conc
        self.n = 0
        self.v = 0
        self.novel = novel          # parent-relative novelty (v1.13 continuity)
        self.new_mask = new_mask    # repertoire-expansion: mask not live at mint time (PRIMARY F1 metric)
        self.consolidated = False


def random_pair(rng, n_ch):
    a, b = rng.sample(range(n_ch), OP_WEIGHT)
    return (1 << a) | (1 << b)


def pattern_channels(pattern, n_ch):
    return [b for b in range(n_ch) if (pattern >> b) & 1]


def initial_pattern(rng, n_ch):
    m = 0
    for c in rng.sample(range(n_ch), PATTERN_WEIGHT):
        m |= 1 << c
    return m


def couples(mask, p):
    """op succeeds iff its channel-set is a SUBSET of the true pattern (v1.13 semantics)."""
    return mask != 0 and (mask & p) == mask


def weight2_subsets(pattern):
    ch = pattern_channels(pattern, 64)
    return [(1 << ch[i]) | (1 << ch[j]) for i in range(len(ch)) for j in range(i + 1, len(ch))]


def step_env(pattern, drift, rng, n_ch):
    """Local random walk on the weight-3 shell: w.p. drift, swap ONE active channel for an inactive one."""
    if rng.random() < drift:
        active = pattern_channels(pattern, n_ch)
        inactive = [b for b in range(n_ch) if not ((pattern >> b) & 1)]
        if active and inactive:
            out = rng.choice(active)
            inn = rng.choice(inactive)
            return (pattern & ~(1 << out)) | (1 << inn)
    return pattern


def sense(pattern, rng, n_ch):
    """Constant-difficulty lossy read: w.p. SENSE_P, misreport ONE channel (swap active->inactive)."""
    if rng.random() < SENSE_P:
        active = pattern_channels(pattern, n_ch)
        inactive = [b for b in range(n_ch) if not ((pattern >> b) & 1)]
        if active and inactive:
            out = rng.choice(active)
            inn = rng.choice(inactive)
            return (pattern & ~(1 << out)) | (1 << inn)
    return pattern


def is_novel(nm, parent_masks, n_ch):
    """Parent-relative novelty on the weight-3 shell: nm couples to SOME weight-3 pattern no parent could.
    nm={x,y} couples to p={x,y,z} for any z; novel iff some such p has no parent as a subset."""
    if nm == 0 or bin(nm).count("1") != OP_WEIGHT:
        return False
    for z in range(n_ch):
        if (nm >> z) & 1:
            continue
        p = nm | (1 << z)
        if not any(couples(pm, p) for pm in parent_masks):
            return True
    return False


def mint_op(ops, rng, n_ch):
    """Weight-2-preserving LOCAL variation (matches local drift). Generic — encodes no pattern."""
    if rng.random() < 0.6:
        parent = rng.choice(ops)
        bits = pattern_channels(parent.mask, n_ch)
        keep = rng.choice(bits)
        others = [b for b in range(n_ch) if b != keep]
        nm = (1 << keep) | (1 << rng.choice(others))
        parents = [parent.mask]
    else:
        if len(ops) >= 2:
            a, c = rng.sample(ops, 2)
            abits = pattern_channels(a.mask, n_ch)
            cbits = pattern_channels(c.mask, n_ch)
            x = rng.choice(abits)
            ypool = [b for b in cbits if b != x]
            if ypool:
                nm = (1 << x) | (1 << rng.choice(ypool))
            else:
                nm = (1 << x) | (1 << rng.choice([b for b in range(n_ch) if b != x]))
            parents = [a.mask, c.mask]
        else:
            nm = random_pair(rng, n_ch)
            parents = []
    novel = is_novel(nm, parents, n_ch)
    new_mask = nm not in {o.mask for o in ops}   # repertoire-expansion at mint time
    return Op(nm, CONC_INIT, novel=novel, new_mask=new_mask)


def run_chemostat(world, drift, seed, adaptive=True):
    n_ch = WORLDS[world]
    rng = random.Random(seed * 1000 + int(drift * 10000) + n_ch)
    pattern = initial_pattern(rng, n_ch)

    ops = [Op(random_pair(rng, n_ch), CONC_INIT) for _ in range(N_BASE_OPS)]
    seen = set()
    uniq = []
    for o in ops:
        if o.mask not in seen:
            seen.add(o.mask)
            uniq.append(o)
    ops = uniq
    # COLD-START GUARANTEE (identical both worlds): ensure >=1 base op couples to the INITIAL pattern.
    if not any(couples(o.mask, pattern) for o in ops):
        subs = weight2_subsets(pattern)
        ops[0] = Op(rng.choice(subs), CONC_INIT)

    budget = B0
    alive = True
    novel_consol_events = []       # v1.13 parent-relative novelty (continuity)
    newmask_consol_events = []     # PRIMARY F1 metric: consolidated repertoire-expansion
    windows = []
    fires = succ = 0
    w_fires = w_succ = w_cover = 0

    for t in range(HORIZON):
        pattern = step_env(pattern, drift, rng, n_ch)
        sensed = sense(pattern, rng, n_ch)

        # functional coverage: can the live repertoire couple to the TRUE pattern right now?
        if any(couples(o.mask, pattern) for o in ops):
            w_cover += 1

        acted = False
        triggered = [o for o in ops if (o.mask & sensed)]
        if triggered:
            if adaptive:
                tot = sum(o.conc for o in triggered)
                x = rng.random() * tot
                acc = 0.0
                chosen = triggered[-1]
                for o in triggered:
                    acc += o.conc
                    if acc >= x:
                        chosen = o
                        break
            else:
                chosen = rng.choice(triggered)   # frozen_random control: uniform firing
            acted = True

        if acted:
            budget -= ACT_COST
            fires += 1
            w_fires += 1
            if couples(chosen.mask, pattern):
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
            for o in ops:
                o.conc *= CONC_DECAY
            ops = [o for o in ops if o.conc >= CONC_MIN]
            if not ops:
                ops = [Op(random_pair(rng, n_ch), CONC_INIT)]

            for o in ops:
                if (not o.consolidated) and o.n >= MIN_N and o.v / o.n >= VIABLE_RATIO:
                    o.consolidated = True
                    if o.novel:
                        novel_consol_events.append(t)
                    if o.new_mask:
                        newmask_consol_events.append(t)

            if rng.random() < VAR_P and len(ops) < MAX_OPS:
                ops.append(mint_op(ops, rng, n_ch))

            if len(ops) > MAX_OPS:
                ops.sort(key=lambda o: o.conc)
                ops = ops[-MAX_OPS:]

        if budget <= 0:
            alive = False
            budget = 0.0
            windows.append({
                "t": t, "budget": budget, "V": len(ops),
                "coverage": w_cover / WINDOW,
                "coupling": (w_succ / w_fires) if w_fires else 0.0,
                "novel_consol_cum": len(novel_consol_events),
                "newmask_consol_cum": len(newmask_consol_events),
                "alive": False,
            })
            break

        if (t + 1) % WINDOW == 0:
            windows.append({
                "t": t + 1, "budget": budget, "V": len(ops),
                "coverage": w_cover / WINDOW,
                "coupling": (w_succ / w_fires) if w_fires else 0.0,
                "novel_consol_cum": len(novel_consol_events),
                "newmask_consol_cum": len(newmask_consol_events),
                "alive": True,
            })
            w_fires = w_succ = w_cover = 0

    coupling_rate = (succ / fires) if fires else 0.0
    return {
        "world": world, "n_ch": n_ch, "drift": drift, "seed": seed, "adaptive": adaptive,
        "alive": alive, "survival_steps": (windows[-1]["t"] if windows else 0),
        "coupling_rate": coupling_rate,
        "novel_consol_total": len(novel_consol_events),
        "newmask_consol_total": len(newmask_consol_events),
        "novel_consol_events": novel_consol_events,
        "newmask_consol_events": newmask_consol_events,
        "final_V": len(ops),
        "windows": windows,
    }


# ============================ analysis / provisional judgment ============================
def _rate_in_window(events, lo, hi, n_windows):
    c = sum(1 for e in events if lo <= e < hi)
    return c / n_windows if n_windows else 0.0


def cell_metrics(runs):
    """Aggregate the 4 seeds of one (world, drift) cell into prereg metrics."""
    n = len(runs)
    survival = sum(1 for r in runs if r["alive"]) / n
    alive_runs = [r for r in runs if r["alive"]]

    # second-half coverage (median across alive seeds); fall back to all seeds if none alive
    def cov2(r):
        sh = [w["coverage"] for w in r["windows"] if w["t"] > HORIZON // 2]
        return sum(sh) / len(sh) if sh else 0.0
    cov_pool = alive_runs if alive_runs else runs
    covs = sorted(cov2(r) for r in cov_pool)
    coverage_2h = covs[len(covs) // 2] if covs else 0.0

    # new-mask minting tail ratio: second-half rate vs first-productive-quarter rate
    q_windows = (HORIZON // 4) // WINDOW      # 10 windows in first quarter
    h_windows = (HORIZON // 2) // WINDOW      # 20 windows in second half
    tails = []
    for r in (alive_runs if alive_runs else runs):
        fq = _rate_in_window(r["newmask_consol_events"], 0, HORIZON // 4, q_windows)
        sh = _rate_in_window(r["newmask_consol_events"], HORIZON // 2, HORIZON, h_windows)
        tails.append((sh / fq) if fq > 0 else None)
    known = [x for x in tails if x is not None]
    mint_tail = (sum(known) / len(known)) if known else None

    excess = sum(r["coupling_rate"] - r["background_rho"] for r in runs) / n
    newmask_total = sum(r["newmask_consol_total"] for r in runs) / n

    # feasible-volume second-half normalized slope (steady-state class)
    def slope2(r):
        vs = [w["V"] for w in r["windows"] if w["t"] > HORIZON // 2]
        if len(vs) < 2:
            return 0.0
        mean = sum(vs) / len(vs)
        if mean == 0:
            return 0.0
        xs = list(range(len(vs)))
        mx = sum(xs) / len(xs)
        num = sum((x - mx) * (v - mean) for x, v in zip(xs, vs))
        den = sum((x - mx) ** 2 for x in xs)
        return (num / den) / mean if den else 0.0
    slopes = [slope2(r) for r in (alive_runs if alive_runs else runs)]
    slope = sum(slopes) / len(slopes) if slopes else 0.0

    return {
        "survival": survival, "coverage_2h": coverage_2h, "mint_tail": mint_tail,
        "excess": excess, "newmask_total": newmask_total, "slope_2h": slope,
        "n_alive": len(alive_runs),
    }


def judge(cells):
    """Apply the pre-registered judgments. cells[(world,drift)] = metrics."""
    def get(w, d):
        return cells.get((w, d))

    # P-F1-SMALL-SATURATES: at SMALL surviving drifts, coverage>=cut AND mint frozen
    small_sat = []
    for d in DRIFTS:
        m = get("SMALL", d)
        if m and m["survival"] > 0 and m["coverage_2h"] >= SATURATION_CUT and \
           (m["mint_tail"] is not None and m["mint_tail"] < FROZEN_TAIL_RATIO):
            small_sat.append(d)
    P_SMALL = len(small_sat) > 0

    # P-F1-OPEN-SUSTAINS: exists OPEN drift with survival>=0.9, coverage<cut, mint>=0.5, excess>=margin
    open_sustain = []
    for d in DRIFTS:
        m = get("OPEN", d)
        if m and m["survival"] >= SURVIVAL_HIGH and m["coverage_2h"] < SATURATION_CUT and \
           (m["mint_tail"] is not None and m["mint_tail"] >= SUSTAIN_TAIL_RATIO) and \
           m["excess"] >= EXCESS_MARGIN:
            open_sustain.append(d)
    P_OPEN = len(open_sustain) > 0

    # P-CONTRAST: at matched drift in the OPEN sustaining interval, OPEN beats SMALL on both axes
    P_CONTRAST = False
    for d in open_sustain:
        so, ss = get("OPEN", d), get("SMALL", d)
        if so and ss:
            to = so["mint_tail"] if so["mint_tail"] is not None else -1
            ts = ss["mint_tail"] if ss["mint_tail"] is not None else -1
            if to > ts and (1 - so["coverage_2h"]) > (1 - ss["coverage_2h"]):
                P_CONTRAST = True
                break

    # P-OC12: OPEN at drift=0 freezes (no environment door)
    m0 = get("OPEN", 0.0)
    P_OC12 = bool(m0 and (m0["mint_tail"] is None or m0["mint_tail"] < FROZEN_TAIL_RATIO or
                          m0["newmask_total"] < 1.0))

    open_all_die = all(get("OPEN", d) and get("OPEN", d)["survival"] <= SURVIVAL_COLLAPSE for d in DRIFTS)
    open_surv_frozen = any(get("OPEN", d) and get("OPEN", d)["survival"] >= SURVIVAL_HIGH and
                           (get("OPEN", d)["mint_tail"] is None or get("OPEN", d)["mint_tail"] < FROZEN_TAIL_RATIO) and
                           get("OPEN", d)["coverage_2h"] < SATURATION_CUT for d in DRIFTS)
    small_nosat = any(get("SMALL", d) and get("SMALL", d)["survival"] > 0 and
                      get("SMALL", d)["coverage_2h"] < SATURATION_CUT for d in DRIFTS)
    open_sat = any(get("OPEN", d) and get("OPEN", d)["coverage_2h"] >= SATURATION_CUT for d in DRIFTS)

    if open_all_die:
        H = "H-OPEN-DIES"
    elif P_OPEN and P_SMALL and P_CONTRAST:
        H = "H-F1-LIFTED"
    elif open_surv_frozen:
        H = "H-STILL-FROZEN"
    elif (small_nosat and not P_SMALL) or open_sat:
        H = "H-NO-ISOLATION"
    else:
        H = "H-PARTIAL/AMBIGUOUS"

    return {
        "H": H,
        "P-F1-SMALL-SATURATES": P_SMALL, "saturating_drifts": small_sat,
        "P-F1-OPEN-SUSTAINS": P_OPEN, "sustaining_drifts": open_sustain,
        "P-CONTRAST": P_CONTRAST, "P-OC12": P_OC12,
        "open_all_die": open_all_die, "open_surv_frozen": open_surv_frozen,
        "small_nosat": small_nosat, "open_sat": open_sat,
    }


def main():
    out = []
    for world in WORLDS:
        for drift in DRIFTS:
            for seed in SEEDS:
                ad = run_chemostat(world, drift, seed, adaptive=True)
                fz = run_chemostat(world, drift, seed, adaptive=False)
                ad["background_rho"] = fz["coupling_rate"]
                ad["excess_causal_validity"] = ad["coupling_rate"] - fz["coupling_rate"]
                ad["frozen_alive"] = fz["alive"]
                out.append(ad)
                mt = ad["newmask_consol_total"]
                print(f"{world:<5} drift={drift:<5} seed={seed} alive={str(ad['alive']):<5} "
                      f"surv={ad['survival_steps']:<5} V={ad['final_V']:<3} "
                      f"newmask={mt:<3} novel={ad['novel_consol_total']:<3} "
                      f"coupling={ad['coupling_rate']:.3f} rho={fz['coupling_rate']:.3f} "
                      f"excess={ad['excess_causal_validity']:+.3f}")

    # per-cell aggregation + provisional judgment
    cells = {}
    for world in WORLDS:
        for drift in DRIFTS:
            runs = [r for r in out if r["world"] == world and r["drift"] == drift]
            cells[(world, drift)] = cell_metrics(runs)

    print("\n================= PER-CELL METRICS (median/mean over 4 seeds) =================")
    print(f"{'world':<6}{'drift':<7}{'surv':<6}{'cov2h':<7}{'mint_tail':<11}{'excess':<9}{'newmask':<9}{'slope2h':<9}")
    for world in WORLDS:
        for drift in DRIFTS:
            m = cells[(world, drift)]
            mt = "None" if m["mint_tail"] is None else f"{m['mint_tail']:.2f}"
            print(f"{world:<6}{drift:<7}{m['survival']:<6.2f}{m['coverage_2h']:<7.3f}"
                  f"{mt:<11}{m['excess']:<+9.3f}{m['newmask_total']:<9.1f}{m['slope_2h']:<+9.3f}")

    verdict = judge(cells)
    print("\n================= PROVISIONAL JUDGMENT (audit against prereg) =================")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    with open(os.path.join(_HERE, "v116_results.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary = {
        "cells": {f"{w}|{d}": cells[(w, d)] for w in WORLDS for d in DRIFTS},
        "verdict": verdict,
    }
    with open(os.path.join(_HERE, "v116_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(out)} runs to v116_results.jsonl and v116_summary.json")


if __name__ == "__main__":
    main()
