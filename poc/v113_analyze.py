"""Local analysis of v113_results.jsonl — no new simulation, just honest reading."""
import json

runs = [json.loads(l) for l in open("v113_results.jsonl", encoding="utf-8")]

def q(series, frac):
    if not series:
        return float("nan")
    k = max(1, int(len(series) * frac))
    return sum(series[-k:]) / k

print(f"{'drift':>6} {'seed':>4} {'alive':>5} {'surv':>5} {'die_t':>6} "
      f"{'V_first':>7} {'V_last':>6} {'cov_sat_t':>9} {'nov_early':>9} {'nov_late':>8} {'excess':>7}")
by_drift = {}
for r in runs:
    w = r["windows"]
    HZ = 4000
    die_t = HZ if r["alive"] else (w[-1]["t"] if w else 0)
    Vs = [x["V"] for x in w]
    V_first = sum(Vs[:max(1,len(Vs)//10)])/max(1,len(Vs)//10) if Vs else 0
    V_last = q(Vs, 0.1)
    # time to saturation (channels_covered == 6)
    sat_t = next((x["t"] for x in w if x.get("channels_covered", 0) >= 6), None)
    ev = r["novel_consol_events"]
    nov_early = sum(1 for e in ev if e < HZ*0.25)
    nov_late = sum(1 for e in ev if e > HZ*0.75)
    d = r["drift"]
    by_drift.setdefault(d, []).append(r)
    print(f"{d:>6} {r['seed']:>4} {str(r['alive']):>5} {r['survival_steps']:>5} {die_t:>6} "
          f"{V_first:>7.1f} {V_last:>6.1f} {str(sat_t):>9} {nov_early:>9} {nov_late:>8} "
          f"{r['excess_causal_validity']:>+7.3f}")

print("\n=== per-drift summary ===")
print(f"{'drift':>6} {'surv_rate':>9} {'mean_excess':>11} {'mean_novel':>10} "
      f"{'novel_late_frac':>15} {'sat_by_1000':>11}")
for d in sorted(by_drift):
    rs = by_drift[d]
    sr = sum(1 for r in rs if r["alive"])/len(rs)
    me = sum(r["excess_causal_validity"] for r in rs)/len(rs)
    mn = sum(r["novel_consol_total"] for r in rs)/len(rs)
    # fraction of novel consolidations happening in last 25% (sustained vs early-burst)
    tot_ev = sum(len(r["novel_consol_events"]) for r in rs)
    late_ev = sum(1 for r in rs for e in r["novel_consol_events"] if e > 4000*0.75)
    lf = late_ev/tot_ev if tot_ev else 0
    satn = sum(1 for r in rs for x in r["windows"] if x["t"]<=1000 and x.get("channels_covered",0)>=6)
    sat_by = sum(1 for r in rs if any(x.get("channels_covered",0)>=6 for x in r["windows"] if x["t"]<=1000))/len(rs)
    print(f"{d:>6} {sr:>9.2f} {me:>+11.3f} {mn:>10.1f} {lf:>15.2f} {sat_by:>11.2f}")
