"""Win-expectancy model: P(team100 wins | per-minute state diff).
Group-split by match (no leakage), side-symmetric via mirroring. Foundation for linear weights."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

FEATURES = ["minute", "gold_diff", "xp_diff", "cs_diff", "level_diff", "tower_diff",
            "inhib_diff", "plate_diff", "dragon_diff", "baron_diff", "herald_diff",
            "grub_diff", "kill_diff"]

panel = pd.read_parquet("data/parsed/panel.parquet")
panel = panel[panel.minute >= 1]  # minute 0 = no info

# group split FIRST (on real matches), then mirror only within each split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(panel, groups=panel.match_id))
train, test = panel.iloc[tr_idx], panel.iloc[te_idx]
print(f"train matches={train.match_id.nunique():,} rows={len(train):,} | test matches={test.match_id.nunique():,} rows={len(test):,}")

def mirror(df):
    m = df.copy()
    for c in FEATURES:
        if c != "minute":
            m[c] = -m[c]
    m["label"] = 1 - m["label"]
    return pd.concat([df, m], ignore_index=True)

train_m = mirror(train)
Xtr, ytr = train_m[FEATURES], train_m["label"]
Xte, yte = test[FEATURES], test["label"]

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6,
                                     l2_regularization=1.0, random_state=42)
clf.fit(Xtr, ytr)
joblib.dump({"model": clf, "features": FEATURES}, "data/win_model.joblib")

p = clf.predict_proba(Xte)[:, 1]
print(f"\nOVERALL test AUC={roc_auc_score(yte, p):.4f}  Brier={brier_score_loss(yte, p):.4f}")

print("\n== AUC & calibration by game minute ==")
print(f"{'min':>5} {'n':>7} {'AUC':>7} {'mean_p':>7} {'actual':>7}")
for lo, hi in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 99)]:
    seg = test[(test.minute >= lo) & (test.minute < hi)]
    if len(seg) < 50 or seg.label.nunique() < 2:
        continue
    ps = clf.predict_proba(seg[FEATURES])[:, 1]
    print(f"{lo:>2}-{hi:<2} {len(seg):>7,} {roc_auc_score(seg.label, ps):>7.3f} {ps.mean():>7.3f} {seg.label.mean():>7.3f}")

print("\n== calibration (test, decile bins) ==")
cal = pd.DataFrame({"p": p, "y": yte.values})
cal["bin"] = pd.qcut(cal.p, 10, duplicates="drop")
tab = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
print(tab.round(3).to_string())

print("\n== win-expectancy table: P(win) vs gold lead (other diffs = 0) ==")
print(f"{'gold_diff':>9} | " + " ".join(f"min{mm:>2}" for mm in [10, 15, 20, 25, 30]))
for gd in [-8000, -4000, -2000, 0, 2000, 4000, 8000]:
    cells = []
    for mm in [10, 15, 20, 25, 30]:
        row = {f: 0 for f in FEATURES}; row["minute"] = mm; row["gold_diff"] = gd
        cells.append(f"{clf.predict_proba(pd.DataFrame([row])[FEATURES])[0,1]:>5.2f}")
    print(f"{gd:>9} | " + "   ".join(cells))
