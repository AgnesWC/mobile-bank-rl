"""Live demonstration.

    python demo.py                 # train (about a minute), then show the schedule
    python demo.py --years 500     # shorter training for a quick run
    python demo.py --load          # reuse the table saved by the last run
    python demo.py --seed 3        # a different district

Steps shown:
  1. the district the agent faces (hidden from the agent)
  2. training, with the test-style return printed every 500 years
  3. one 28-day cycle of the learned schedule, day by day
  4. the Q-values behind the first decision
  5. comparison with the baselines on unseen years
"""
import argparse
import json
import os
import time

import numpy as np

from mobilebank import config as C
from mobilebank.env import MobileBankEnv, decode
from mobilebank.agents import QLearning
from mobilebank.baselines import (RandomPolicy, FixedRotation, NearestOverdue,
                                  CalendarPlanner)
from mobilebank.runner import train, evaluate, summarise
from mobilebank.shield import CoverageShield

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--years", type=int, default=4000)
ap.add_argument("--load", action="store_true")
ap.add_argument("--cycle", type=int, default=5, help="cycle of the year to show (0-12)")
args = ap.parse_args()

hp = dict(alpha=0.02, decay_years=1200)
if os.path.exists("results/chosen_hp.json"):
    hp = json.load(open("results/chosen_hp.json"))
hp["decay_years"] = min(hp["decay_years"], int(0.3 * args.years))

env = MobileBankEnv()
env.reset(seed=args.seed)
params = env.params

print("\n1. THE DISTRICT (hidden from the agent)")
print(f"{'site':19s}{'base demand/day':>16s}{'account uptake':>16s}{'incident risk/day':>19s}{'under-served':>14s}")
for i, name in enumerate(C.SITES):
    print(f"{name:19s}{params['lam0'][i]:16.1f}{params['uptake'][i]:16.3f}"
          f"{params['rho'][i] * C.RHO_TRAINING_SCALE:19.4f}{'yes' if C.UNDERSERVED[i] else 'no':>14s}")

path = f"results/q_table_seed{args.seed}.npy"
agent = QLearning(seed=args.seed, **hp)
if args.load and os.path.exists(path):
    agent.Q = np.load(path)
    agent.N[:] = 10
    print(f"\n2. TRAINING skipped: loaded {path}")
else:
    print(f"\n2. TRAINING  Q-learning, {args.years} simulated years "
          f"({args.years * 312:,} decisions), step-size floor {hp['alpha']}, "
          f"epsilon 1.0 -> 0.05 over {hp['decay_years']} years")
    env.reset(seed=args.seed + 1000, params=params)
    t0, done = time.time(), 0
    while done < args.years:
        n = min(500, args.years - done)
        train(agent, env, n, start_year=done)
        done += n
        val = summarise(evaluate(agent, params, 20000 + args.seed, 10)[0])["ret"]
        print(f"   year {done:5d}   epsilon {agent.eps:.2f}   "
              f"greedy return per cycle on held-out years: {val:8.1f}")
    print(f"   trained in {time.time() - t0:.0f} s; "
          f"{(agent.N.sum(axis=1) > 0).sum():,} of 9,720 states visited")
    np.save(path, agent.Q)

print(f"\n3. ONE 28-DAY CYCLE OF THE LEARNED SCHEDULE (cycle {args.cycle} of the year)")
env = MobileBankEnv()
s, info = env.reset(seed=30000 + args.seed, params=params)
for _ in range(args.cycle * C.OP_DAYS):               # run the policy up to the chosen cycle
    s, _, _, _, info = env.step(agent.act(s, info, explore=False))
first_state = s
season = ["rainy", "marketing", "lean"][info["season"]]
print(f"   season: {season}")
print(f"   {'day':>4s} {'wk':>3s} {'road':>5s}  {'site':18s}{'km':>5s}{'served':>8s}{'turned away':>13s}"
      f"{'accounts':>10s}{'incident':>10s}{'reward':>9s}")
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
total = 0.0
while True:
    day, wk, road = info["day"], info["week"] + 1, ("wet" if info["wet"] else "dry")
    a = agent.act(s, info, explore=False)
    s, r, _, truncated, info = env.step(a)
    total += r
    print(f"   {names[day % 7]:>4s} {wk:3d} {road:>5s}  {C.SITES[a]:18s}{info['km']:5.0f}{info['served']:8d}"
          f"{info['turned_away']:13d}{info['accounts']:10d}{'YES' if info['incident'] else '':>10s}{r:9.1f}")
    if truncated:
        break
print(f"   net reward for the cycle: {total:.1f}")

print("\n4. THE LOOKUP BEHIND THE FIRST DECISION OF THAT CYCLE")
d = decode(first_state)
others = [C.SITES[j] for j in range(5) if j != d["loc"]]
label = ["fresh", "due", "overdue"]
print(f"   state row {first_state}: at {C.SITES[d['loc']]}, week {d['week'] + 1}, "
      f"{'wet' if d['wet'] else 'dry'} road; "
      + ", ".join(f"{n} {label[b]}" for n, b in zip(others, d["buckets"])))
for i, name in enumerate(C.SITES):
    mark = "  <- chosen" if i == int(np.argmax(agent.Q[first_state])) else ""
    print(f"   Q[{name:18s}] = {agent.Q[first_state, i]:9.1f}{mark}")

print("\n5. COMPARISON ON 20 UNSEEN YEARS (mean per 28-day cycle)")
print(f"   {'policy':27s}{'net reward':>12s}{'worst 5%':>10s}{'incidents':>11s}"
      f"{'under-served visits':>21s}{'longest gap (days)':>20s}")
for pol in [agent, CoverageShield(agent, 7), CalendarPlanner(), NearestOverdue(), FixedRotation(), RandomPolicy(args.seed)]:
    m = summarise(evaluate(pol, params, 30000 + args.seed, 20)[0])
    print(f"   {pol.name:27s}{m['ret']:12.1f}{m['cvar5']:10.1f}{m['incidents']:11.2f}"
          f"{m['under_share']:20.0%} {m['max_gap_overall']:19d}")