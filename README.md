# Reinforcement Learning for Daily Site Scheduling of a Rural Mobile Bank Unit

NMAI511 Reinforcement Learning, Assignment B 
Dione Karuru (263115) and Agnes Chuchu (170235), MSc Artificial Intelligence. Lecturer: S. H. Muchinenyika

A tabular Q-learning agent decides each morning which of five rural growth points one mobile bank unit should serve. The simulator is a custom Gymnasium environment; no RL library is used.

- **Google Colab (run it online):** https://colab.research.google.com/drive/1OByCudojqhW2gRnf9nxp3q0HsL_ABt-W
- **Dataset used for calibration:** [Wisabi Bank Dataset on Kaggle](https://www.kaggle.com/datasets/obinnaiheanachor/wisabi-bank-dataset) (CC0)
- **Presentation:** `RL_Mobile_Bank_Scheduling_Assignment_B_Presentation_updated.pptx`

## Addressing Assignment B

| Assignment requirements | Where |
| --- | --- |
| Method of choosing the best policy (lookup table, 9,720 states x 5 sites) | Notebook Section 5; `mobilebank/agents.py` (`QLearning`, `greedy`) |
| Justification of the RL algorithm (Q-learning, not DQN or PPO) | Notebook Section 3; slide 5 |
| Training algorithm | `mobilebank/runner.py` (`train`), `mobilebank/agents.py` (`update`); notebook Section 4 |
| Evaluation and stability checks before deployment | `mobilebank/runner.py` (`evaluate`), `experiments.py`, `analyze.py`; notebook Sections 6 and 7 (deployment gate) |
| Demonstration | `python demo.py --load` |
| Source code and datasets | This repository; `data/calibration.json` holds the numbers taken from the Wisabi data |

## Layout

```
mobilebank/       simulator (env.py), learners (agents.py), baselines, oracle, coverage rule (shield.py), config.py, runner.py
data/             calibration.json: summary numbers from the Wisabi Bank dataset
results/          final 20-district results (main.json, extras.json, supp.json), validation grid, chosen settings,
                  trained table (q_table_seed0.npy) and figures/
notebooks/        the Colab notebook with every output saved
tests.py          20 unit tests against hand-worked cases
demo.py           live demonstration
experiments.py    the full study (about 1 hour on 2 cores)
supplementary.py  coverage-rule rows (about 1 hour)
analyze.py        prints every number used in the presentation
calibrate.py      re-creates data/calibration.json from the raw Kaggle files
```

## Run it

```bash
pip install -r requirements.txt
python -m unittest tests -v         # 20 tests
python demo.py --load               # uses the saved table results/q_table_seed0.npy
python demo.py                      # trains for 4,000 simulated years first (1 to 2 minutes)
python analyze.py                   # prints the results from results/*.json
```

Reproducing the study:

```bash
python experiments.py --part grid     # validation grid for the step-size floor and epsilon decay
python experiments.py                 # 20 districts, 4,000 training years each
python supplementary.py               # coverage-rule rows
python analyze.py
```

`python experiments.py --quick` runs 2 districts and 300 years as a smoke test and writes to `results/quick/`, so it never overwrites the full results.

## Data

The Wisabi Bank Dataset (2.1 million ATM transactions, 50 locations) sets the spread of daily demand (variance 3.2 times the mean) and 4.14 minutes per transaction. Only these summary numbers are kept, in `data/calibration.json`.Site demand levels, seasonal peaks, account uptake and incident risk are labelled assumptions in `mobilebank/config.py`.

## Main results (20 districts, 20 unseen test years each)

Mean net reward per 28-day cycle: oracle 473, Q-learning 261, Expected SARSA 231, nearest overdue 179, calendar planner 145, fixed rotation -130, random -347. Q-learning beats the calendar planner in 20 of 20 districts. The table alone leaves some under-served wards waiting up to 164 days, so the deployed policy adds a coverage rule (visit an under-served ward after 7 days), which brings the longest wait to 10 days. See the presentation and notebook Section 7 for the full checks.
