"""Unit tests against hand-calculated cases.   Run:  python -m unittest tests -v"""
import unittest

import numpy as np

from mobilebank import config as C
from mobilebank.env import MobileBankEnv, n_states, decode, bucket
from mobilebank.agents import QLearning, ExpectedSarsa
from mobilebank.oracle import Oracle


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = MobileBankEnv(rho_scale=0.0)       # no incidents: rewards are easy to check
        self.s, self.info = self.env.reset(seed=7)

    def test_state_count(self):
        self.assertEqual(n_states("3"), 5 * 81 * 4 * 3 * 2)
        self.assertEqual(n_states("3"), 9720)

    def test_state_index_round_trip(self):
        for _ in range(200):
            s, _, _, _, info = self.env.step(self.env.action_space.sample())
            d = decode(s)
            self.assertEqual(d["loc"], info["loc"])
            self.assertEqual(d["week"], info["week"])
            self.assertEqual(d["season"], info["season"])
            self.assertEqual(d["wet"], int(info["wet"]))
            others = [j for j in range(5) if j != info["loc"]]
            self.assertEqual(d["buckets"], [bucket(info["ages"][j]) for j in others])

    def test_buckets(self):
        self.assertEqual([bucket(a) for a in (1, 2, 3, 6, 7, 30)], [0, 0, 1, 1, 2, 2])

    def test_cycle_has_24_decisions_and_skips_sunday(self):
        steps, days = 0, []
        while True:
            days.append(self.env.day_in_cycle)
            _, _, terminated, truncated, _ = self.env.step(0)
            steps += 1
            self.assertFalse(terminated)              # the task never terminates
            if truncated:
                break
        self.assertEqual(steps, 24)
        self.assertEqual(self.env.t_abs, 28)
        self.assertTrue(all(d % 7 != 6 for d in days))  # no decision on a Sunday

    def test_distances(self):
        self.assertTrue(np.allclose(C.DIST, C.DIST.T))
        self.assertEqual(C.DIST[0, 3], 35)            # A-D, Assignment A
        self.assertEqual(C.DIST[1, 4], 105)           # B-E
        self.assertEqual((C.DIST > 0).sum(), 20)      # all 10 pairs set, both directions

    def test_reward_matches_hand_calculation(self):
        for a in (1, 4, 4, 0, 2, 3):
            _, r, _, _, i = self.env.step(a)
            expected = (C.W_N * i["served"] + C.W_A * i["accounts"] + i["bonus"]
                        - C.W_T * i["turned_away"] - C.C_KM * i["km"] - C.C_DAY)
            self.assertAlmostEqual(r, expected)

    def test_bonus_rules(self):
        _, _, _, _, i = self.env.step(1)              # tobacco ward, waited 8 days, under-served
        self.assertEqual(i["bonus"], C.W_U)
        _, _, _, _, i = self.env.step(1)              # staying: no bonus
        self.assertEqual(i["bonus"], 0.0)
        _, _, _, _, i = self.env.step(0)              # the district town is not under-served
        self.assertEqual(i["bonus"], 0.0)
        _, _, _, _, i = self.env.step(1)              # back after 2 days: not yet due
        self.assertEqual(i["bonus"], 0.0)

    def test_capacity(self):
        env = MobileBankEnv(rho_scale=0.0, lambda_scale=5.0)
        env.reset(seed=1)
        _, _, _, _, i = env.step(3)                   # mining settlement, overdue: demand far above 120
        self.assertEqual(i["served"], C.CAPACITY_TX)
        self.assertGreater(i["turned_away"], 0)
        self.assertLessEqual(i["accounts"], C.CAPACITY_ACC)

    def test_demand_distribution(self):
        # negative binomial: mean mu, variance mu + mu^2 / k (k measured on the Wisabi dataset)
        rng = np.random.default_rng(0)
        mu, k = 60.0, C.DEMAND_K
        x = rng.negative_binomial(k, k / (k + mu), size=200000)
        self.assertAlmostEqual(x.mean(), mu, delta=0.3)
        self.assertAlmostEqual(x.var(), mu + mu**2 / k, delta=3.0)
        self.assertEqual(len(C.WEEKDAY_MULT), 6)
        self.assertAlmostEqual(C.WEEKDAY_MULT.mean(), 1.0, places=3)

    def test_pent_up_factor(self):
        self.assertEqual(C.psi(1, True), 0.3)
        self.assertEqual([C.psi(a, False) for a in (2, 3, 6, 7)], [0.7, 1.0, 1.0, 1.3])

    def test_same_seed_same_district(self):
        e1, e2 = MobileBankEnv(), MobileBankEnv()
        e1.reset(seed=3)
        e2.reset(seed=3)
        out1 = [e1.step(a)[1] for a in (1, 2, 3, 4, 0)]
        out2 = [e2.step(a)[1] for a in (1, 2, 3, 4, 0)]
        self.assertEqual(out1, out2)


class TestLearners(unittest.TestCase):
    def test_q_learning_update_by_hand(self):
        ag = QLearning(alpha=0.005, gamma=0.95)
        ag.Q[20] = [0, 10, 0, 0, 0]
        ag.update(10, 2, 5.0, 20, {})
        # first visit: step size 1/1, target = 5 + 0.95 * 10 = 14.5
        self.assertAlmostEqual(ag.Q[10, 2], 14.5)
        ag.update(10, 2, 1.0, 20, {})
        # second visit: step size 1/2, target = 1 + 9.5 = 10.5 -> 14.5 + 0.5 * (10.5 - 14.5)
        self.assertAlmostEqual(ag.Q[10, 2], 12.5)
        self.assertEqual(ag.N[10, 2], 2)

    def test_step_size_floor(self):
        ag = QLearning(alpha=0.005)
        ag.N[0, 0] = 10
        self.assertAlmostEqual(ag.lr(0, 0), 0.1)
        ag.N[0, 0] = 1000
        self.assertAlmostEqual(ag.lr(0, 0), 0.005)

    def test_expected_sarsa_update_by_hand(self):
        ag = ExpectedSarsa(alpha=0.005, gamma=0.95)
        ag.eps = 0.5
        ag.Q[20] = [0, 10, 0, 0, 0]
        ag.update(10, 2, 5.0, 20, {})
        # policy at s'=20: 0.1 each + 0.5 on the best -> expected value 0.6 * 10 = 6
        self.assertAlmostEqual(ag.Q[10, 2], 5 + 0.95 * 6)

    def test_greedy_and_epsilon(self):
        ag = QLearning(seed=0)
        ag.Q[5] = [1, 2, 9, 4, 0]
        info = dict(loc=0, wet=False, week=0)
        self.assertEqual(ag.act(5, info, explore=False), 2)
        ag.eps = 0.0
        self.assertEqual(ag.act(5, info, explore=True), 2)
        ag.set_year(0)
        self.assertEqual(ag.eps, 1.0)
        ag.set_year(10**6)
        self.assertAlmostEqual(ag.eps, 0.05)

    def test_safety_mask(self):
        ag = QLearning(seed=0)
        # wet morning at the tobacco ward: the irrigation scheme is gravel and 105 km away
        self.assertNotIn(4, ag.allowed_explore(dict(loc=1, wet=True, week=0)))
        # wet morning at the district town: it is 70 km away -> allowed
        self.assertIn(4, ag.allowed_explore(dict(loc=0, wet=True, week=0)))
        # dry morning: nothing is masked
        self.assertEqual(len(ag.allowed_explore(dict(loc=1, wet=False, week=0))), 5)
        # month-end week: the site with the highest estimated incident rate is excluded
        ag.inc[2] += 5
        self.assertNotIn(2, ag.allowed_explore(dict(loc=0, wet=False, week=3)))
        # the mask as written in Assignment A also blocks the tobacco ward in the marketing season
        strict = QLearning(seed=0, safe="proposal")
        self.assertNotIn(1, strict.allowed_explore(dict(loc=0, wet=False, week=0, season=1)))
        self.assertIn(1, strict.allowed_explore(dict(loc=0, wet=False, week=0, season=0)))

    def test_calendar_and_risk_rules(self):
        self.assertTrue(C.high_risk(0, 3, 0))         # any site in month-end week
        self.assertTrue(C.high_risk(1, 0, 1))         # tobacco ward in the marketing season
        self.assertFalse(C.high_risk(4, 0, 2))
        self.assertTrue(C.in_season_peak(4, 2))       # irrigation scheme peaks in the lean season
        self.assertFalse(C.in_season_peak(1, 2))
        self.assertEqual(C.p_wet(0, True), 0.6)
        self.assertEqual(C.p_wet(0, False), 0.25)
        self.assertEqual(C.p_wet(4, False), 0.05)     # March
        self.assertEqual(C.p_wet(10, True), 0.0)      # never in the lean season


class TestCoverageRule(unittest.TestCase):
    def test_rule_overrides_only_when_a_ward_is_due(self):
        from mobilebank.shield import CoverageShield
        ag = QLearning(seed=0)
        ag.Q[:] = [9, 0, 0, 0, 0]                      # the table always prefers the district town
        rule = CoverageShield(ag, limit=7)
        self.assertEqual(rule.act(0, dict(loc=0, ages=np.array([1, 3, 6, 9, 2]))), 0)
        self.assertEqual(rule.act(0, dict(loc=0, ages=np.array([1, 3, 8, 9, 7]))), 2)   # longest-waiting ward
        self.assertEqual(rule.overrides, 1)

    def test_rule_keeps_gaps_within_10_days(self):
        from mobilebank.shield import CoverageShield
        from mobilebank.runner import evaluate, summarise
        ag = QLearning(seed=0)
        ag.Q[:] = [9, 0, 0, 0, 0]
        env = MobileBankEnv()
        env.reset(seed=5)
        rows, _, _ = evaluate(CoverageShield(ag, 7), env.params, 11, 2)
        self.assertLessEqual(summarise(rows)["max_gap_overall"], 10)


class TestOracle(unittest.TestCase):
    def test_oracle_day_index(self):
        o = Oracle.__new__(Oracle)                    # no need to solve for this check
        info = dict(loc=0, ages=np.array([1, 3, 5, 7, 9]), day=8, wet=False)
        # day 8 = Tuesday of week 2 = operating day 7
        self.assertEqual((o.index(info) // 2) % C.OP_DAYS, 7)


if __name__ == "__main__":
    unittest.main()