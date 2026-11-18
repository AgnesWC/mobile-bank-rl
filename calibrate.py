"""Calibrate the shape of daily demand from the Wisabi Bank dataset (Kaggle, CC0).

    python calibrate.py /path/to/wisabi_folder

Reads the five *_transactions.csv files and writes data/calibration.json:
weekday multipliers, dispersion of daily counts and minutes per transaction.
Only these summary numbers are kept in the project; the raw data is not copied.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

folder = sys.argv[1]
frames = []
for f in sorted(glob.glob(os.path.join(folder, "*_transactions.csv"))):
    d = pd.read_csv(f, encoding="utf-8-sig")
    d["state"] = os.path.basename(f).split("_")[0]
    frames.append(d)
tx = pd.concat(frames, ignore_index=True)
tx["start"] = pd.to_datetime(tx["TransactionStartDateTime"], format="%m/%d/%Y %H:%M")
tx["end"] = pd.to_datetime(tx["TransactionEndDateTime"], format="%m/%d/%Y %H:%M")
tx["date"] = tx["start"].dt.normalize()
tx["minutes"] = (tx["end"] - tx["start"]).dt.total_seconds() / 60
print(f"transactions: {len(tx):,}; locations: {tx['LocationID'].nunique()}; "
      f"dates: {tx['date'].min().date()} to {tx['date'].max().date()}")
types = {1: "Withdrawal", 2: "Deposit", 3: "Balance Inquiry", 4: "Transfer"}
print("type mix:", (tx["TransactionTypeID"].map(types).value_counts(normalize=True) * 100).round(1).to_dict())
print("minutes per transaction: mean %.2f, median %.1f, by type %s" % (
    tx["minutes"].mean(), tx["minutes"].median(),
    tx.groupby(tx["TransactionTypeID"].map(types))["minutes"].mean().round(2).to_dict()))
hour = tx["start"].dt.hour + tx["start"].dt.minute / 60
print("share of transactions between 08:00 and 16:30: %.1f%%" % (100 * ((hour >= 8) & (hour < 16.5)).mean()))

# daily counts per location
daily = tx.groupby(["LocationID", "date"]).size().rename("n").reset_index()
daily["dow"] = daily["date"].dt.dayofweek            # 0 = Monday
daily["dom"] = daily["date"].dt.day
daily["month"] = daily["date"].dt.month
daily["loc_mean"] = daily.groupby("LocationID")["n"].transform("mean")
daily["rel"] = daily["n"] / daily["loc_mean"]
print(f"location-days: {len(daily):,}; mean daily count per location: "
      f"min {daily['loc_mean'].min():.0f}, median {daily['loc_mean'].median():.0f}, max {daily['loc_mean'].max():.0f}")

dow = daily.groupby("dow")["rel"].mean()
print("weekday multipliers (Mon..Sun):", dow.round(3).tolist())
daily["wom"] = np.minimum((daily["dom"] - 1) // 7, 3)   # 0..3, 3 = day 22 to month-end
print("week-of-month multipliers:", daily.groupby("wom")["rel"].mean().round(3).tolist())
last5 = daily["date"].dt.days_in_month - daily["dom"] < 5
print("last 5 days of month vs rest: %.3f vs %.3f" % (daily.loc[last5, "rel"].mean(), daily.loc[~last5, "rel"].mean()))
print("month multipliers:", daily.groupby("month")["rel"].mean().round(3).tolist())

# dispersion after removing the location mean and weekday effect
daily["expected"] = daily["loc_mean"] * daily["dow"].map(dow)
g = daily.groupby("LocationID")
ratio = g.apply(lambda d: ((d["n"] - d["expected"]) ** 2).mean() / d["expected"].mean(), include_groups=False)
print("variance / mean of daily counts by location: median %.2f, range %.2f to %.2f" % (
    ratio.median(), ratio.min(), ratio.max()))
cv = g.apply(lambda d: (d["n"] / d["expected"]).std(), include_groups=False)
print("coefficient of variation of daily counts: median %.3f" % cv.median())
# extra-Poisson variance: Var = mu + mu^2 / k  ->  k = mu^2 / (Var - mu)
mu = g["expected"].mean()
var = g.apply(lambda d: ((d["n"] - d["expected"]) ** 2).mean(), include_groups=False)
k = (mu ** 2 / (var - mu)).where(var > mu)
print("negative-binomial size k by location: median %.1f (locations with Var > mean: %d of %d)" % (
    k.median(), k.notna().sum(), len(k)))

mon_sat = dow.loc[0:5] / dow.loc[0:5].mean()
out = dict(source="Wisabi Bank Dataset, Kaggle (obinnaiheanachor), CC0: Public Domain",
           transactions=int(len(tx)), locations=int(tx["LocationID"].nunique()),
           weekday_multiplier_mon_to_sat=[round(float(x), 4) for x in mon_sat],
           dispersion_k=round(float(k.median()), 2) if k.notna().any() else None,
           variance_to_mean=round(float(ratio.median()), 3),
           minutes_per_transaction=round(float(tx["minutes"].mean()), 3),
           share_in_operating_hours=round(float(((hour >= 8) & (hour < 16.5)).mean()), 4))
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, "data", "calibration.json"), "w"), indent=1)
print(json.dumps(out, indent=1))