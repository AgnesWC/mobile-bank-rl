"""Print every number used in the presentation from results/*.json.

    python analyze.py                # the full run in results/
    python analyze.py results/quick  # the quick run
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from mobilebank.baselines import CalendarPlanner, NearestOverdue
from mobilebank.runner import evaluate, summarise
from experiments import district, VAL_YEARS

D = sys.argv[1] if len(sys.argv) > 1 else "results"
main = json.load(open(f"{D}/main.json"))
n = len(main)


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def col(policy, key):
    return np.array([m["test"][policy][key] for m in main], float)


print(f"districts: {n}; chosen hp: {json.load(open(D + '/chosen_hp.json'))}")
grid = json.load(open(f"{D}/grid.json"))
by = {}
for g in grid:
    by.setdefault((g["alpha"], g["decay"]), []).append(g["val"])
print("\nVALIDATION GRID (mean greedy return per cycle, districts 100+)")
for k, v in sorted(by.items()):
    print(f"  alpha {k[0]:<6} decay {k[1]:<5} mean {np.mean(v):7.1f}   per district {np.round(v)}")

print("\nTEST RESULTS (mean over districts, +/- 95% CI half-width)")
names = ["Oracle", "Q-learning", "Q-learning (eps 0.05)", "Expected SARSA", "Nearest overdue",
         "Calendar planner", "Fixed rotation", "Random", "Greedy average"]
hdr = f"{'policy':24s}{'return':>9s}{'ci':>6s}{'surplus':>9s}{'cvar5':>8s}{'served':>8s}{'turned':>8s}{'accts':>7s}{'km':>7s}{'incid':>7s}{'under%':>8s}{'maxgap(mean/worst)':>20s}{'entropy':>9s}"
print(hdr)
for p in names:
    r = col(p, "ret")
    ent = np.mean([m["test"][p].get("entropy", np.nan) for m in main])
    print(f"{p:24s}{r.mean():9.1f}{ci(r):6.1f}{col(p,'surplus').mean():9.1f}{col(p,'cvar5').mean():8.0f}"
          f"{col(p,'served').mean():8.0f}{col(p,'turned').mean():8.1f}{col(p,'accounts').mean():7.1f}"
          f"{col(p,'km').mean():7.0f}{col(p,'incidents').mean():7.3f}{100*col(p,'under_share').mean():8.1f}"
          f"{col(p,'max_gap_overall').mean():11.1f}/{col(p,'max_gap_overall').max():<6.0f}{ent:9.2f}")

print("\nPAIRED DIFFERENCES, Q-learning minus baseline (per district)")
q = col("Q-learning", "ret")
for p in ["Calendar planner", "Nearest overdue", "Fixed rotation", "Expected SARSA", "Random"]:
    d = q - col(p, "ret")
    t = stats.ttest_rel(q, col(p, "ret"))
    w = stats.ttest_ind(q, col(p, "ret"), equal_var=False)
    print(f"  vs {p:18s} diff {d.mean():7.1f} +/- {ci(d):5.1f}   wins {int((d > 0).sum())}/{n}   "
          f"paired t p={t.pvalue:.4f}   Welch p={w.pvalue:.4f}   relative {100*d.mean()/abs(col(p,'ret').mean()):.1f}%")
print(f"  across-district sd of Q-learning return: {q.std(ddof=1):.1f}")

print("\nSUCCESS CRITERIA")
print(f"  1. return: Q {q.mean():.1f} vs fixed rotation {col('Fixed rotation','ret').mean():.1f}, calendar {col('Calendar planner','ret').mean():.1f}")
g = col("Q-learning", "max_gap_overall")
print(f"  2. longest gap for an under-served ward: mean {g.mean():.1f}, worst {g.max():.0f} days; districts with gap <= 10: {(g <= 10).sum()}/{n}")
print(f"     baselines worst gap: calendar {col('Calendar planner','max_gap_overall').max():.0f}, rotation {col('Fixed rotation','max_gap_overall').max():.0f}, nearest {col('Nearest overdue','max_gap_overall').max():.0f}")
qi, fi = col("Q-learning", "incidents"), col("Fixed rotation", "incidents")
print(f"  3. incidents per cycle: Q {qi.mean():.3f} vs fixed rotation {fi.mean():.3f} (diff {np.mean(qi-fi):.3f} +/- {ci(qi-fi):.3f})")
fq = np.array([m["fixed_demand"]["Q-learning"] for m in main])
fo = np.array([m["fixed_demand"]["Oracle"] for m in main])
fc = np.array([m["fixed_demand"]["Calendar planner"] for m in main])
fn = np.array([m["fixed_demand"]["Nearest overdue"] for m in main])
print(f"  4. fixed demand: Q {fq.mean():.1f}, oracle {fo.mean():.1f}, ratio of means {100*fq.mean()/fo.mean():.1f}%; "
      f"calendar {100*fc.mean()/fo.mean():.1f}%, nearest {100*fn.mean()/fo.mean():.1f}%; districts with Q >= 90% of oracle: {(fq >= 0.9*fo).sum()}/{n}")
print(f"     drifting demand: Q/oracle ratio of means {100*q.mean()/col('Oracle','ret').mean():.1f}%; regret per cycle {np.mean(col('Oracle','ret')-q):.1f}")

print("\nSTABILITY")
vc = np.array([[v for _, v in m["q_val_curve"]] for m in main])
ec = np.array([[v for _, v in m["es_val_curve"]] for m in main])
yrs = [y for y, _ in main[0]["q_val_curve"]]
print("  held-out greedy return by training year (mean over districts):")
for i, y in enumerate(yrs):
    if y in (100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000) or i == len(yrs) - 1:
        print(f"    year {y:5d}   Q {vc[:, i].mean():7.1f}   ES {ec[:, i].mean():7.1f}")
k = max(1, len(yrs) // 4)
last, prev = vc[:, -k:].mean(axis=1), vc[:, -2 * k:-k].mean(axis=1)
print(f"  last quarter of training vs the quarter before: {last.mean():.1f} vs {prev.mean():.1f} "
      f"(change {np.mean(last-prev):.1f} +/- {ci(last-prev):.1f}, {100*np.mean(last-prev)/prev.mean():.1f}%)")
print(f"  sd of the held-out return over the last quarter's checkpoints, mean over districts: {vc[:, -k:].std(axis=1, ddof=1).mean():.1f}")
ch = np.array([[c for _, c in m["q_churn"]] for m in main])
print(f"  policy churn (share of visited states whose greedy action changed in 100 years): "
      f"first {100*ch[:,0].mean():.1f}%, middle {100*ch[:,ch.shape[1]//2].mean():.1f}%, last {100*ch[:,-1].mean():.1f}%")
print(f"  visited states: mean {np.mean([m['visited_states'] for m in main]):.0f} of 9720; "
      f"test decisions in entries updated >= 10 times: {100*np.mean([m['q_well_visited'] for m in main]):.1f}% (min {100*np.min([m['q_well_visited'] for m in main]):.1f}%)")
print(f"  table finite in all districts: {all(m['q_finite'] for m in main)}; largest |Q|: {max(m['q_absmax'] for m in main):.0f}")
tr = np.array([m["q_curve"] for m in main])
print(f"  training return (with exploration), mean of years 1-100: {tr[:, :100].mean():.1f}; last 100: {tr[:, -100:].mean():.1f}")
print(f"  eps kept at 0.05 in deployment: {col('Q-learning (eps 0.05)','ret').mean():.1f} vs greedy {q.mean():.1f}")

print("\nSTRESS TESTS (policy trained on the base simulator)")
for case in main[0]["stress"]:
    line = f"  {case:24s}"
    vals = {}
    for p in ["Q-learning", "Calendar planner", "Nearest overdue"]:
        v = np.array([m["stress"][case][p]["ret"] for m in main])
        vals[p] = v
        line += f"{p[:8]} {v.mean():7.1f}  "
    d = vals["Q-learning"] - vals["Calendar planner"]
    d2 = vals["Q-learning"] - vals["Nearest overdue"]
    inc = np.mean([m["stress"][case]["Q-learning"]["incidents"] for m in main])
    gap = np.max([m["stress"][case]["Q-learning"]["max_gap_overall"] for m in main])
    print(line + f"Q-cal {d.mean():6.1f}+/-{ci(d):5.1f} wins {int((d>0).sum())}/{n}  Q-near {d2.mean():6.1f} wins {int((d2>0).sum())}/{n}  inc {inc:.3f} worst gap {gap}")

print("\nPREDICTABILITY (softmax over learned Q at evaluation)")
print(f"  greedy: return {q.mean():.1f}, entropy {np.mean([m['test']['Q-learning']['entropy'] for m in main]):.2f} bits; "
      f"calendar planner entropy {np.mean([m['test']['Calendar planner']['entropy'] for m in main]):.2f}; "
      f"fixed rotation {np.mean([m['test']['Fixed rotation']['entropy'] for m in main]):.2f}; max possible {np.log2(5):.2f}")
for t in main[0]["softmax"]:
    r = np.array([m["softmax"][t]["ret"] for m in main])
    e = np.array([m["softmax"][t]["entropy"] for m in main])
    print(f"  temperature {t:>3}: return {r.mean():7.1f} ({100*(r.mean()-q.mean())/q.mean():+.1f}%), entropy {e.mean():.2f} bits")

print("\nSITING ACCURACY")
print(f"  Kendall tau mean {np.mean([m['siting']['tau'] for m in main]):.2f}; top-two match {sum(m['siting']['top2'] for m in main)}/{n}")

print("\nTRAINING-TIME RISK while epsilon decays")
for p in ["Q-learning", "Expected SARSA"]:
    inc = np.array([m["train_risk"][p]["incidents"] for m in main])
    rk = np.array([m["train_risk"][p]["risky_trips"] for m in main])
    rt = np.array([m["train_risk"][p]["train_return"] for m in main])
    print(f"  {p:16s} incidents {inc.mean():8.1f}   peak-day trips {rk.mean():9.1f}   mean training return {rt.mean():7.1f}")
di = np.array([m["train_risk"]["Expected SARSA"]["risky_trips"] - m["train_risk"]["Q-learning"]["risky_trips"] for m in main])
print(f"  ES minus Q peak-day trips: {di.mean():.1f} +/- {ci(di):.1f}")

if os.path.exists(f"{D}/extras.json"):
    ex = json.load(open(f"{D}/extras.json"))
    seeds = [e["seed"] for e in ex]
    base = np.array([main[s]["test"]["Q-learning"]["ret"] for s in seeds])
    print(f"\nEXTRAS on districts {seeds} (Q-learning on these: {base.mean():.1f})")
    for k in ["Q-learning (mask on)", "No safety mask", "Mask as in Assignment A", "Monte Carlo", "Count bonus", "R2 (overdue penalty)", "w_U=0", "w_U=80", "w_U=120"]:
        t = [e[k]["test"] for e in ex]
        line = (f"  {k:22s} return {np.mean([x['ret'] for x in t]):7.1f}  surplus {np.mean([x['surplus'] for x in t]):7.1f}  "
                f"under {100*np.mean([x['under_share'] for x in t]):5.1f}%  worst gap {max(x['max_gap_overall'] for x in t):4d}  "
                f"mean gap {np.mean([x['max_gap_overall'] for x in t]):5.1f}  incidents {np.mean([x['incidents'] for x in t]):.3f}")
        if "risk" in ex[0][k]:
            line += f"  training incidents {np.mean([e[k]['risk']['incidents'] for e in ex]):.0f}  train return {np.mean([e[k]['risk']['train_return'] for e in ex]):.1f}"
        print(line)
    q40 = [main[s]["test"]["Q-learning"] for s in seeds]
    print(f"  {'w_U=40 (main run)':22s} return {np.mean([x['ret'] for x in q40]):7.1f}  surplus {np.mean([x['surplus'] for x in q40]):7.1f}  "
          f"under {100*np.mean([x['under_share'] for x in q40]):5.1f}%  worst gap {max(x['max_gap_overall'] for x in q40):4d}  mean gap {np.mean([x['max_gap_overall'] for x in q40]):5.1f}")
    print("  bandit layer, mean return per cycle over the first 5 years:")
    for k in ex[0]["bandits"]:
        print(f"    {k:24s}{np.mean([e['bandits'][k] for e in ex]):8.1f}")
    for nm in ["Calendar planner", "Fixed rotation", "Random"]:
        print(f"    ({nm} on test years, same districts: {np.mean([main[s]['test'][nm]['ret'] for s in seeds]):.1f})")

if os.path.exists(f"{D}/supp.json"):
    sp = json.load(open(f"{D}/supp.json"))
    print("\nSUPPLEMENTARY (5 districts)")
    for k in sp[0]:
        if k != "seed":
            print(f"  {k:42s}{np.mean([s[k] for s in sp]):9.1f}")

# baselines on the validation years, for the learning-curve reference lines
cal, near = [], []
for m in main:
    p = district(m["seed"])
    cal.append(summarise(evaluate(CalendarPlanner(), p, 20000 + m["seed"], VAL_YEARS)[0])["ret"])
    near.append(summarise(evaluate(NearestOverdue(), p, 20000 + m["seed"], VAL_YEARS)[0])["ret"])
print(f"\nvalidation-year references: calendar planner {np.mean(cal):.1f}, nearest overdue {np.mean(near):.1f}")
print("curve rows:", json.dumps([dict(year=y, q=round(float(vc[:, i].mean())), es=round(float(ec[:, i].mean()))) for i, y in enumerate(yrs)]))