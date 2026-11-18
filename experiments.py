"""Run every experiment in the report.

    python experiments.py            # full run (20 districts), about 1 hour on 2 cores
    python experiments.py --quick    # 2 districts, short training, for a check

Writes results/*.json (the --quick run writes to results/quick/ so that it never
overwrites the results of a full run).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import kendalltau

from mobilebank import config as C
from mobilebank.env import MobileBankEnv
from mobilebank.agents import QLearning, ExpectedSarsa, MonteCarlo, CountBonusQ, Bandit
from mobilebank.baselines import (RandomPolicy, FixedRotation, GreedyAverage,
                                  NearestOverdue, CalendarPlanner)
from mobilebank.oracle import Oracle
from mobilebank.runner import train, evaluate, summarise, schedule_entropy

TEST_YEARS = 20
VAL_YEARS = 10
CHECK_EVERY = 100


def district(seed):
    env = MobileBankEnv()
    env.reset(seed=seed)
    return env.params


def fit(Agent, params, seed, years, env_kwargs=None, val_seed=None, **agent_kw):
    """Train one agent; record a learning curve and policy snapshots."""
    env = MobileBankEnv(**(env_kwargs or {}))
    env.reset(seed=seed + 1000, params=params)          # training random stream
    agent = Agent(seed=seed, **agent_kw)
    curve, val_curve, snaps = [], [], {}
    inc, risky = [], []
    done = 0
    while done < years:
        n = min(CHECK_EVERY, years - done)
        log, _ = train(agent, env, n, start_year=done)
        done += n
        curve += list(np.array(log["cycle_return"]).reshape(n, 13).mean(axis=1))
        inc += list(np.array(log["cycle_incidents"]).reshape(n, 13).sum(axis=1))
        risky += list(np.array(log["risky_trips"]).reshape(n, 13).sum(axis=1))
        snaps[done] = agent.Q.argmax(axis=1).copy()
        if val_seed is not None:
            rows, _, _ = evaluate(agent, params, val_seed, VAL_YEARS, env_kwargs)
            val_curve.append((done, summarise(rows)["ret"]))
    k = int(min(agent.decay_years, years))
    risk = dict(incidents=float(np.sum(inc[:k])), risky_trips=float(np.sum(risky[:k])),
                train_return=float(np.mean(curve[:k])))
    return agent, dict(curve=curve, val_curve=val_curve, snaps=snaps, risk=risk)


def true_ranking(params):
    """Expected base demand over a year, with the mean calendar peaks."""
    lam = params["lam0"].copy()
    lam[0] *= (1 + 0.25 * (C.PEAK_MONTH_END[0] - 1))
    lam[2] *= (1 + 0.25 * (C.PEAK_MONTH_END[2] - 1))
    for i, (season, mult) in C.PEAK_SEASON.items():
        share = float(np.mean(C.SEASON_OF_CYCLE == season))
        lam[i] *= (1 + share * (mult - 1))
    return lam


def churn(snaps, visited):
    ks = sorted(snaps)
    out = []
    for a, b in zip(ks[:-1], ks[1:]):
        out.append((b, float(np.mean(snaps[a][visited] != snaps[b][visited]))))
    return out


def main_job(args):
    seed, years, quick, hp = args
    t0 = time.time()
    params = district(seed)
    val_seed, test_seed = 20000 + seed, 30000 + seed
    res = dict(seed=seed, params={k: v.tolist() for k, v in params.items()})

    q, qlog = fit(QLearning, params, seed, years, val_seed=val_seed, **hp)
    es, eslog = fit(ExpectedSarsa, params, seed, years, val_seed=val_seed, **hp)
    visited = q.N.sum(axis=1) > 0
    res["q_curve"], res["q_val_curve"] = qlog["curve"], qlog["val_curve"]
    res["es_curve"], res["es_val_curve"] = eslog["curve"], eslog["val_curve"]
    res["q_churn"] = churn(qlog["snaps"], visited)
    res["q_finite"] = bool(np.isfinite(q.Q).all())
    res["q_absmax"] = float(np.abs(q.Q).max())
    res["visited_states"] = int(visited.sum())

    oracle = Oracle(params)
    policies = {"Q-learning": q, "Expected SARSA": es, "Random": RandomPolicy(seed),
                "Fixed rotation": FixedRotation(), "Greedy average": GreedyAverage(),
                "Nearest overdue": NearestOverdue(), "Calendar planner": CalendarPlanner(),
                "Oracle": oracle}
    test = {}
    for name, pol in policies.items():
        rows, choices, states = evaluate(pol, params, test_seed, TEST_YEARS)
        s = summarise(rows)
        s["entropy"] = schedule_entropy(choices)
        s["cycle_returns"] = [r["ret"] for r in rows]
        test[name] = s
        if name == "Q-learning":
            st = np.array(states)
            acts = np.array([c[2] for c in choices])
            res["q_well_visited"] = float(np.mean(q.N[st, acts] >= 10))
    rows, _, _ = evaluate(q, params, test_seed, TEST_YEARS, explore=True)   # eps = 0.05 kept
    test["Q-learning (eps 0.05)"] = summarise(rows)
    res["test"] = test

    # fixed demand: no drift, compare with the oracle
    fixed = {}
    for name in ["Q-learning", "Calendar planner", "Nearest overdue", "Oracle"]:
        rows, _, _ = evaluate(policies[name], params, test_seed, TEST_YEARS, dict(drift=False))
        fixed[name] = summarise(rows)["ret"]
    res["fixed_demand"] = fixed

    # stress tests (policy trained on the base simulator, tested on a shifted one)
    stress_cases = {"demand -20%": dict(lambda_scale=0.8), "demand +20%": dict(lambda_scale=1.2),
                    "wet days x2": dict(wet_scale=2.0), "realistic incident rate": dict(rho_scale=1.0),
                    "incident rate x2": dict(rho_scale=2 * C.RHO_TRAINING_SCALE)}
    stress = {}
    for case, kw in stress_cases.items():
        stress[case] = {}
        for name in ["Q-learning", "Calendar planner", "Nearest overdue"]:
            rows, _, _ = evaluate(policies[name], params, test_seed, TEST_YEARS, kw)
            stress[case][name] = summarise(rows)
    res["stress"] = stress

    # predictability: softmax evaluation policy on the learned Q
    soft = {}
    for temp in [5, 10, 20, 40]:
        rows, choices, _ = evaluate(q, params, test_seed, TEST_YEARS, softmax_temp=temp)
        s = summarise(rows)
        soft[temp] = dict(ret=s["ret"], entropy=schedule_entropy(choices),
                          incidents=s["incidents"])
    res["softmax"] = soft

    # siting accuracy: rank sites by mean transactions served per visit (visit records)
    est = np.array([np.mean(r) if r else 0.0 for r in q.records])
    truth = true_ranking(params)
    res["siting"] = dict(tau=float(kendalltau(est, truth).statistic),
                         top2=bool(set(np.argsort(-est)[:2]) == set(np.argsort(-truth)[:2])),
                         visits=[len(r) for r in q.records])

    # training-time risk while epsilon decays (Q-learning vs Expected SARSA)
    res["train_risk"] = {"Q-learning": qlog["risk"], "Expected SARSA": eslog["risk"]}
    res["minutes"] = (time.time() - t0) / 60
    print(f"seed {seed} done in {res['minutes']:.1f} min", flush=True)
    return res


def extras_job(args):
    """Comparisons run on fewer districts: MC, count bonus, reward design,
    bonus size, safety mask, bandit layer."""
    seed, years, hp = args
    params = district(seed)
    val_seed, test_seed = 20000 + seed, 30000 + seed
    out = dict(seed=seed)

    def test_of(agent, env_kwargs=None):
        rows, _, _ = evaluate(agent, params, test_seed, TEST_YEARS, env_kwargs)
        return summarise(rows)

    mc, mclog = fit(MonteCarlo, params, seed, years, **hp)
    cb, cblog = fit(CountBonusQ, params, seed, years, alpha=hp["alpha"])
    out["Monte Carlo"] = dict(test=test_of(mc), curve=mclog["curve"])
    out["Count bonus"] = dict(test=test_of(cb), curve=cblog["curve"])
    masked, mlog = fit(QLearning, params, seed, years, **hp)
    out["Q-learning (mask on)"] = dict(test=test_of(masked), risk=mlog["risk"])
    nomask, nmlog = fit(QLearning, params, seed, years, safe=False, **hp)
    out["No safety mask"] = dict(test=test_of(nomask), risk=nmlog["risk"])
    strict, stlog = fit(QLearning, params, seed, years, safe="proposal", **hp)
    out["Mask as in Assignment A"] = dict(test=test_of(strict), risk=stlog["risk"])
    r2env = dict(reward_design="R2")
    r2, _ = fit(QLearning, params, seed, years, env_kwargs=r2env, **hp)
    out["R2 (overdue penalty)"] = dict(test=test_of(r2))     # scored on the R1 simulator
    for wu in [0, 80, 120]:
        ag, _ = fit(QLearning, params, seed, years, env_kwargs=dict(w_u=wu), **hp)
        out[f"w_U={wu}"] = dict(test=test_of(ag))
    # bandit layer: 5 simulated years, no learning of state
    band = {}
    for label, kw in {"eps-greedy 0.05": dict(kind="egreedy", eps=0.05),
                      "eps-greedy 0.1": dict(kind="egreedy", eps=0.1),
                      "eps-greedy 0.3": dict(kind="egreedy", eps=0.3),
                      "optimistic (init 300)": dict(kind="optimistic", eps=0.0, init=300),
                      "softmax (50 to 5)": dict(kind="softmax", horizon=5 * 312),
                      "UCB1 (c=2)": dict(kind="ucb", c=2.0),
                      "Discounted UCB (0.97)": dict(kind="ducb", c=2.0, disc=0.97)}.items():
        env = MobileBankEnv()
        env.reset(seed=test_seed, params=params)
        b = Bandit(seed=seed, **kw)
        total = 0.0
        for _ in range(5 * 312):
            a = b.act()
            _, r, _, _, _ = env.step(a)
            b.update(a, r)
            total += r
        band[label] = total / (5 * 13)        # mean return per cycle
    out["bandits"] = band
    print(f"extras seed {seed} done", flush=True)
    return out


def grid_job(args):
    """Validation grid for the step-size floor alpha and the epsilon decay length."""
    seed, alpha, decay, years = args
    params = district(seed)
    agent, _ = fit(QLearning, params, seed, years, alpha=alpha, decay_years=decay)
    rows, _, _ = evaluate(agent, params, 20000 + seed, VAL_YEARS)
    return dict(seed=seed, alpha=alpha, decay=decay, val=summarise(rows)["ret"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", default="all", choices=["all", "grid", "main", "extras"])
    a = ap.parse_args()
    RES = "results/quick" if a.quick else "results"
    os.makedirs(RES, exist_ok=True)
    years = 300 if a.quick else 4000
    n_seeds, n_extra = (2, 1) if a.quick else (20, 5)
    with Pool(2) as pool:
        if a.part in ("all", "grid"):
            decays = (90, 150) if a.quick else (600, 1200)
            jobs = [(s, al, d, years) for s in range(100, 100 + (1 if a.quick else 3))
                    for al in (0.005, 0.02) for d in decays]
            grid = pool.map(grid_job, jobs)
            json.dump(grid, open(f"{RES}/grid.json", "w"), indent=1)
        # pick the configuration with the best mean validation return
        grid = json.load(open(f"{RES}/grid.json"))
        best = {}
        for g in grid:
            best.setdefault((g["alpha"], g["decay"]), []).append(g["val"])
        alpha, decay = max(best, key=lambda k: np.mean(best[k]))
        hp = dict(alpha=alpha, decay_years=decay)
        print("chosen on validation:", hp, flush=True)
        json.dump(hp, open(f"{RES}/chosen_hp.json", "w"))
        if a.part in ("all", "main"):
            main = pool.map(main_job, [(s, years, a.quick, hp) for s in range(n_seeds)])
            json.dump(main, open(f"{RES}/main.json", "w"))
        if a.part in ("all", "extras"):
            extras = pool.map(extras_job, [(s, years, hp) for s in range(n_extra)])
            json.dump(extras, open(f"{RES}/extras.json", "w"))