"""Supplementary runs on 5 districts (run after experiments.py):

  - the settings proposed in Assignment A (alpha = 0.1 constant, epsilon
    decayed over 4 years, 6 training years)
  - alpha = 0.1 constant with the full 4,000-year training run
  - a finer state with four waiting-time buckets (30,720 states)
  - wall-clock training time for the chosen configuration
  - the coverage rule (mobilebank/shield.py) applied to the trained table
"""
import json
import time
from multiprocessing import Pool

from experiments import district, fit, TEST_YEARS
from mobilebank.agents import QLearning
from mobilebank.shield import CoverageShield
from mobilebank.runner import evaluate, summarise


YEARS = 4000


def job(seed):
    hp = json.load(open("results/chosen_hp.json"))
    params = district(seed)
    test_seed = 30000 + seed

    def score(agent):
        return summarise(evaluate(agent, params, test_seed, TEST_YEARS)[0])["ret"]

    out = dict(seed=seed)
    a, _ = fit(QLearning, params, seed, 6, alpha=0.1, harmonic=False, decay_years=4)
    out["Assignment A settings, 6 years"] = score(a)
    a, _ = fit(QLearning, params, seed, 6, **hp)
    out["final settings, 6 years"] = score(a)
    a, _ = fit(QLearning, params, seed, YEARS, alpha=0.1, harmonic=False,
               decay_years=hp["decay_years"])
    out["alpha 0.1 constant, 4000 years"] = score(a)
    t0 = time.time()
    a, _ = fit(QLearning, params, seed, YEARS, **hp)
    out["train_seconds"] = time.time() - t0
    out["final settings, 4000 years"] = score(a)
    plain = summarise(evaluate(a, params, test_seed, TEST_YEARS)[0])
    out["plain: worst gap"] = plain["max_gap_overall"]
    out["plain: under-served share %"] = 100 * plain["under_share"]
    out["plain: incidents"] = plain["incidents"]
    for limit in (7, 9):
        sh = CoverageShield(a, limit)
        m = summarise(evaluate(sh, params, test_seed, TEST_YEARS)[0])
        out[f"shield {limit}: return"] = m["ret"]
        out[f"shield {limit}: worst gap"] = m["max_gap_overall"]
        out[f"shield {limit}: under-served share %"] = 100 * m["under_share"]
        out[f"shield {limit}: incidents"] = m["incidents"]
        out[f"shield {limit}: override share %"] = 100 * sh.overrides / sh.calls
    a, _ = fit(QLearning, params, seed, YEARS, bins="4", env_kwargs=dict(bins="4"), **hp)
    out["four buckets (30,720 states), 4000 years"] = score(a)
    out["four buckets visited"] = int((a.N.sum(axis=1) > 0).sum())
    print("supp seed", seed, "done", flush=True)
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(job, range(5))
    json.dump(res, open("results/supp.json", "w"), indent=1)