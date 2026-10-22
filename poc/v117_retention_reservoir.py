"""
v1.17 — attack the v1.16 openness<->trackability WALL: does a RETENTION-DEFAULT reservoir
(no acute execution of a partially-matching coupler) dissolve it?
See poc/preregister/v117_retention_reservoir.yaml (COMMITTED BEFORE RESULTS, 5f01f93).

Non-teleological frame inherited verbatim from v1.13 / v1.16. The agent has NO goal, NO
target, NO oracle, and does NOT know its future. viability = self-maintenance (budget>0).
capability crystallizes as a BYPRODUCT of staying coupled to a drifting flux. Measured by
OC-9 excess + survival + coverage + feasible volume + minting continuity + RESERVOIR DIVERSITY,
NEVER win-rate.

SINGLE swept structural variable = the ACUTE PUNISHMENT on a FAILED fire (monotone 3-rung ladder):
  ACUTE     = 0.6 (= CONC_INIT): ONE failed fire deletes the coupler.        [v1.16 baseline]
  SOFT      = 0.1: ONE failed fire survives with margin (0.6->0.5 >> 0.3); ~3 failures delete.
  RESERVOIR = 0.0: NO acute execution; a coupler is removed ONLY by the GLOBAL dissipative decay
              (conc *= CONC_DECAY=0.997/step) when unreinforced (~231 steps 0.6->0.3). This is the
              pure bet-hedging reservoir: use-it-or-lose-it dissipation PRESERVED [DS-C2], acute
              execution REMOVED.

POSITIVE JUSTIFICATION IS L0/L1, NOT BIOLOGY (biology = hypothesis source only, K-BIOLOGY-NOT-PROOF):
  v1.16's single-failure execution violates P2.1 (default-retain), P2-B1 (no un-audited mass
  deletion), C2.3 (no single-op silent deletion of a candidate pivot), FS-C2/FS-C4 (feasible-space
  shrinkage = degeneration). DS-C2/P1.2 require the reservoir stay dissipative, not a frozen store.

EVERY physical/energy constant, the world structure (weights, local drift walk, constant-difficulty
sensing, local variation, cold-start), the drift grid and the seeds are HELD IDENTICAL to v1.16
(which held v1.13). survival threshold coupling = (DECAY+ACT_COST)/GAIN = 0.4333 is UNCHANGED, so
the reservoir grants NO free success: survival still REQUIRES a subset-matching coupler to fire
>= 43.33% of the time. RESERVOIR cannot pass trivially. Physical constants are NOT re-tuned
(K-NO-TUNE-TO-WIN / S4 / G5); the acute-punishment ladder is a pre-registered STRUCTURAL sweep with
a MONOTONE (dose-response) prediction, not a search for a survivable value.
"""
import json
import random
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- physical constants (IDENTICAL to v1.13/v1.16) -------------
DECAY = 1.0
GAIN = 3.0
ACT_COST = 0.3
B0 = 25.0
BUDGET_MAX = 60.0
HORIZON = 4000
WINDOW = 100

CONC_INIT = 0.6
CONC_UP = 1.0
CONC_DECAY = 0.997
CONC_MIN = 0.3
MAX_OPS = 40

N_BASE_OPS = 6
VAR_P = 0.20
MIN_N = 12
VIABLE_RATIO = 0.5

# ------------------- v1.16 structural choices (HELD, both worlds) ------------------------
OP_WEIGHT = 2
PATTERN_WEIGHT = 3
SENSE_P = 0.30          # constant-difficulty: ONE sensing error w.p. SENSE_P

WORLDS = {"SMALL": 8, "OPEN": 64}
DRIFTS = [0.0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.40]
SEEDS = [42, 43, 44, 45]

# --------- v1.17 SINGLE swept structural variable: acute punishment on a failed fire ------
# monotone 3-rung ladder (dose-response). ACUTE reproduces v1.16 exactly.
PUNISH_RUNGS = {"ACUTE": 0.6, "SOFT": 0.1, "RESERVOIR": 0.0}

# ------------------- frozen judgment thresholds (from prereg, == v1.13/v1.16) --------------
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
    """op succeeds iff its channel-set is a SUBSET of the true pattern (v1.13/v1.16 semantics)."""
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
    """Parent-relative novelty on the weight-3 shell: nm couples to SOME weight-3 pattern no parent could."""
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


def run_chemostat(world, drift, seed, adaptive=True, punish=0.6):
    """punish = acute conc penalty applied on a FAILED fire (the v1.17 swept rung).
    punish=0.6 reproduces v1.16 ACUTE exactly; punish=0.0 is the retention-default RESERVOIR."""
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
    # COLD-START GUARANTEE (identical both worlds, all rungs): >=1 base op couples to the INITIAL pattern.
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
    w_distinct = 0.0               # reservoir-diversity instrument (sum of per-step distinct masks)
    w_persister = 0.0              # reservoir-diversity instrument (sum of per-step persister fraction)
    w_steps = 0

    for t in range(HORIZON):
        pattern = step_env(pattern, drift, rng, n_ch)
        sensed = sense(pattern, rng, n_ch)

        # functional coverage: can the live repertoire couple to the TRUE pattern right now?
        if any(couples(o.mask, pattern) for o in ops):
            w_cover += 1

        # reservoir-diversity instruments (per-step, averaged over the window below)
        if ops:
            w_distinct += len({o.mask for o in ops})
            n_persist = sum(1 for o in ops if not couples(o.mask, pattern))
            w_persister += n_persist / len(ops)
        w_steps += 1

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
                    # v1.17 SINGLE CHANGE: acute penalty is the swept rung (was fixed CONC_DOWN=0.6).
                    chosen.conc = max(0.0, chosen.conc - punish)
                    chosen.n += 1

        budget -= DECAY
        if budget > BUDGET_MAX:
            budget = BUDGET_MAX

        if adaptive:
            for o in ops:
                o.conc *= CONC_DECAY            # global dissipation PRESERVED (DS-C2 use-it-or-lose-it)
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
                "coverage": w_cover / max(1, w_steps),
                "coupling": (w_succ / w_fires) if w_fires else 0.0,
                "distinct": w_distinct / max(1, w_steps),
                "persister": w_persister / max(1, w_steps),
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
                "distinct": w_distinct / WINDOW,
                "persister": w_persister / WINDOW,
                "novel_consol_cum": len(novel_consol_events),
                "newmask_consol_cum": len(newmask_consol_events),
                "alive": True,
            })
            w_fires = w_succ = w_cover = 0
            w_distinct = w_persister = 0.0

    coupling_rate = (succ / fires) if fires else 0.0
    return {
        "world": world, "n_ch": n_ch, "drift": drift, "seed": seed, "adaptive": adaptive,
        "punish": punish,
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
    """Aggregate the 4 seeds of one (world, rung, drift) cell into prereg metrics."""
    n = len(runs)
    survival = sum(1 for r in runs if r["alive"]) / n
    alive_runs = [r for r in runs if r["alive"]]
    pool = alive_runs if alive_runs else runs

    def cov2(r):
        sh = [w["coverage"] for w in r["windows"] if w["t"] > HORIZON // 2]
        return sum(sh) / len(sh) if sh else 0.0
    covs = sorted(cov2(r) for r in pool)
    coverage_2h = covs[len(covs) // 2] if covs else 0.0

    def mean2(r, key):
        sh = [w[key] for w in r["windows"] if w["t"] > HORIZON // 2]
        return sum(sh) / len(sh) if sh else 0.0
    distinct_2h = sum(mean2(r, "distinct") for r in pool) / len(pool)
    persister_2h = sum(mean2(r, "persister") for r in pool) / len(pool)
    v_final = sum(r["final_V"] for r in runs) / n

    q_windows = (HORIZON // 4) // WINDOW
    h_windows = (HORIZON // 2) // WINDOW
    tails = []
    for r in pool:
        fq = _rate_in_window(r["newmask_consol_events"], 0, HORIZON // 4, q_windows)
        sh = _rate_in_window(r["newmask_consol_events"], HORIZON // 2, HORIZON, h_windows)
        tails.append((sh / fq) if fq > 0 else None)
    known = [x for x in tails if x is not None]
    mint_tail = (sum(known) / len(known)) if known else None

    excess = sum(r["coupling_rate"] - r["background_rho"] for r in runs) / n
    newmask_total = sum(r["newmask_consol_total"] for r in runs) / n

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
    slopes = [slope2(r) for r in pool]
    slope = sum(slopes) / len(slopes) if slopes else 0.0

    # steady-state class (v1.13 classifier + SATURATED)
    if survival <= SURVIVAL_COLLAPSE:
        sclass = "COLLAPSE"
    elif coverage_2h >= SATURATION_CUT and (mint_tail is None or mint_tail < FROZEN_TAIL_RATIO):
        sclass = "SATURATED"          # survives but F1-frozen (the SMALL expected outcome)
    elif slope > SLOPE_EPS:
        sclass = "UNBOUNDED"
    elif slope < -SLOPE_EPS:
        sclass = "DECLINING"
    elif mint_tail is not None and mint_tail >= SUSTAIN_TAIL_RATIO:
        sclass = "SUSTAINED"
    else:
        sclass = "PLATEAU"

    return {
        "survival": survival, "coverage_2h": coverage_2h, "mint_tail": mint_tail,
        "excess": excess, "newmask_total": newmask_total, "slope_2h": slope,
        "distinct_2h": distinct_2h, "persister_2h": persister_2h, "v_final": v_final,
        "n_alive": len(alive_runs), "sclass": sclass,
    }


def judge(cells):
    """Apply the pre-registered judgments. cells[(world,rung,drift)] = metrics."""
    def get(w, r, d):
        return cells.get((w, r, d))

    rungs = ["ACUTE", "SOFT", "RESERVOIR"]

    # ---- P-DOSE-RESPONSE: monotone survival ACUTE<=SOFT<=RESERVOIR + non-decreasing band width
    per_cell_viol = 0
    per_cell_tot = 0
    for w in WORLDS:
        for d in DRIFTS:
            s = [get(w, r, d)["survival"] if get(w, r, d) else None for r in rungs]
            if None in s:
                continue
            per_cell_tot += 1
            if not (s[0] <= s[1] + 1e-9 and s[1] <= s[2] + 1e-9):
                per_cell_viol += 1
    band_width = {}
    for w in WORLDS:
        for r in rungs:
            band_width[(w, r)] = sum(1 for d in DRIFTS
                                     if get(w, r, d) and get(w, r, d)["survival"] >= SURVIVAL_HIGH)
    band_monotone = all(band_width[(w, "ACUTE")] <= band_width[(w, "SOFT")] <= band_width[(w, "RESERVOIR")]
                        for w in WORLDS)
    P_DOSE = band_monotone and (per_cell_tot > 0 and per_cell_viol / per_cell_tot <= 0.25)

    # ---- P-RESERVOIR-SMALL-BAND: exists drift>0 where SMALL RESERVOIR survives but ACUTE died
    small_band = [d for d in DRIFTS if d > 0
                  and get("SMALL", "RESERVOIR", d) and get("SMALL", "ACUTE", d)
                  and get("SMALL", "RESERVOIR", d)["survival"] >= SURVIVAL_HIGH
                  and get("SMALL", "ACUTE", d)["survival"] <= SURVIVAL_COLLAPSE]
    P_SMALL_BAND = len(small_band) > 0

    # ---- P-RESERVOIR-OPEN: OPEN RESERVOIR survives w/ coverage<cut, mint sustained, excess>=margin
    open_sustain = [d for d in DRIFTS
                    if get("OPEN", "RESERVOIR", d)
                    and get("OPEN", "RESERVOIR", d)["survival"] >= SURVIVAL_HIGH
                    and get("OPEN", "RESERVOIR", d)["coverage_2h"] < SATURATION_CUT
                    and (get("OPEN", "RESERVOIR", d)["mint_tail"] is not None
                         and get("OPEN", "RESERVOIR", d)["mint_tail"] >= SUSTAIN_TAIL_RATIO)
                    and get("OPEN", "RESERVOIR", d)["excess"] >= EXCESS_MARGIN]
    P_OPEN = len(open_sustain) > 0

    # ---- P-RESERVOIR-DIVERSITY: RESERVOIR holds a strictly more diverse repertoire than ACUTE
    div_wins = div_tot = 0
    for w in WORLDS:
        for d in DRIFTS:
            if d == 0:
                continue
            ra, rr = get(w, "ACUTE", d), get(w, "RESERVOIR", d)
            if not ra or not rr:
                continue
            div_tot += 1
            if rr["distinct_2h"] > ra["distinct_2h"] + 1e-9 and rr["persister_2h"] > ra["persister_2h"] + 1e-9:
                div_wins += 1
    P_DIVERSITY = div_tot > 0 and div_wins / div_tot >= 0.5

    # ---- P-OC12: at drift=0 every arm freezes minting (both worlds)
    P_OC12 = all(get(w, r, 0.0) and (get(w, r, 0.0)["mint_tail"] is None
                                     or get(w, r, 0.0)["mint_tail"] < FROZEN_TAIL_RATIO
                                     or get(w, r, 0.0)["newmask_total"] < 1.0)
                 for w in WORLDS for r in rungs)

    # ---- K-RETAIN-AUDIT: RESERVOIR must not be an all-dead hoard (V pinned, coverage ~0) or negative excess
    retain_flags = []
    for w in WORLDS:
        for d in DRIFTS:
            m = get(w, "RESERVOIR", d)
            if not m:
                continue
            if m["survival"] > 0:
                if m["v_final"] >= 0.95 * MAX_OPS and m["coverage_2h"] < 0.10:
                    retain_flags.append(f"{w}|{d}:hoard")
                if m["excess"] < 0:
                    retain_flags.append(f"{w}|{d}:neg_excess")
    K_RETAIN_PASS = len(retain_flags) == 0

    # ---- composite states
    open_res_all_die = all(get("OPEN", "RESERVOIR", d) and get("OPEN", "RESERVOIR", d)["survival"] <= SURVIVAL_COLLAPSE
                           for d in DRIFTS if d > 0)
    small_res_all_die = all(get("SMALL", "RESERVOIR", d) and get("SMALL", "RESERVOIR", d)["survival"] <= SURVIVAL_COLLAPSE
                            for d in DRIFTS if d > 0)
    res_survives_all_open = all(get("OPEN", "RESERVOIR", d) and get("OPEN", "RESERVOIR", d)["survival"] >= SURVIVAL_HIGH
                                for d in DRIFTS)

    if not K_RETAIN_PASS:
        H = "H-TRIVIAL-OR-BUG"
    elif P_OPEN and P_DOSE and K_RETAIN_PASS:
        H = "H-RESERVOIR-LIFTS-WALL"
    elif P_SMALL_BAND and open_res_all_die:
        H = "H-RESERVOIR-SMALL-ONLY"
    elif open_res_all_die and small_res_all_die:
        H = "H-NO-RESCUE"
    elif per_cell_tot > 0 and per_cell_viol / per_cell_tot > 0.5:
        H = "H-NO-ISOLATION"
    else:
        H = "H-PARTIAL/AMBIGUOUS"

    return {
        "H": H,
        "P-DOSE-RESPONSE": P_DOSE, "band_monotone": band_monotone,
        "per_cell_monotone_violations": f"{per_cell_viol}/{per_cell_tot}",
        "band_width": {f"{w}|{r}": band_width[(w, r)] for w in WORLDS for r in rungs},
        "P-RESERVOIR-SMALL-BAND": P_SMALL_BAND, "small_band_drifts": small_band,
        "P-RESERVOIR-OPEN": P_OPEN, "open_sustain_drifts": open_sustain,
        "P-RESERVOIR-DIVERSITY": P_DIVERSITY, "diversity_wins": f"{div_wins}/{div_tot}",
        "P-OC12": P_OC12,
        "K-RETAIN-AUDIT-PASS": K_RETAIN_PASS, "retain_flags": retain_flags,
        "open_res_all_die": open_res_all_die, "small_res_all_die": small_res_all_die,
        "res_survives_all_open": res_survives_all_open,
    }


def main():
    out = []
    for world in WORLDS:
        for drift in DRIFTS:
            for seed in SEEDS:
                # frozen control: NO punishment, NO adaptation -> identical across rungs -> run ONCE.
                fz = run_chemostat(world, drift, seed, adaptive=False, punish=0.0)
                rho = fz["coupling_rate"]
                for rung, punish in PUNISH_RUNGS.items():
                    ad = run_chemostat(world, drift, seed, adaptive=True, punish=punish)
                    ad["rung"] = rung
                    ad["background_rho"] = rho
                    ad["excess_causal_validity"] = ad["coupling_rate"] - rho
                    ad["frozen_alive"] = fz["alive"]
                    out.append(ad)
                    print(f"{world:<5} {rung:<9} drift={drift:<5} seed={seed} "
                          f"alive={str(ad['alive']):<5} surv={ad['survival_steps']:<5} "
                          f"V={ad['final_V']:<3} newmask={ad['newmask_consol_total']:<3} "
                          f"coupling={ad['coupling_rate']:.3f} rho={rho:.3f} "
                          f"excess={ad['excess_causal_validity']:+.3f}", flush=True)

    # per-cell aggregation (world, rung, drift) + provisional judgment
    cells = {}
    for world in WORLDS:
        for rung in PUNISH_RUNGS:
            for drift in DRIFTS:
                runs = [r for r in out if r["world"] == world and r["rung"] == rung and r["drift"] == drift]
                cells[(world, rung, drift)] = cell_metrics(runs)

    print("\n============= PER-CELL METRICS (median/mean over 4 seeds) =============")
    print(f"{'world':<6}{'rung':<10}{'drift':<7}{'surv':<6}{'cov2h':<7}{'mint':<7}"
          f"{'excess':<9}{'Vfin':<6}{'dist2h':<8}{'pers2h':<8}{'class':<10}")
    for world in WORLDS:
        for rung in PUNISH_RUNGS:
            for drift in DRIFTS:
                m = cells[(world, rung, drift)]
                mt = "None" if m["mint_tail"] is None else f"{m['mint_tail']:.2f}"
                print(f"{world:<6}{rung:<10}{drift:<7}{m['survival']:<6.2f}{m['coverage_2h']:<7.3f}"
                      f"{mt:<7}{m['excess']:<+9.3f}{m['v_final']:<6.1f}{m['distinct_2h']:<8.2f}"
                      f"{m['persister_2h']:<8.3f}{m['sclass']:<10}")

    verdict = judge(cells)
    print("\n============= PROVISIONAL JUDGMENT (audit against prereg) =============")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    with open(os.path.join(_HERE, "v117_results.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary = {
        "cells": {f"{w}|{r}|{d}": cells[(w, r, d)] for w in WORLDS for r in PUNISH_RUNGS for d in DRIFTS},
        "verdict": verdict,
    }
    with open(os.path.join(_HERE, "v117_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(out)} adaptive runs to v117_results.jsonl and v117_summary.json")


if __name__ == "__main__":
    main()
