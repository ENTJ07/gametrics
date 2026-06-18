"""Win model v2 (role-resolved). Learns role-specific resource values jointly.
Group-split by match, side-symmetric via mirroring."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
panel = pd.read_parquet("data/parsed/panel2.parquet")
panel = panel[panel.minute >= 1]
# gold-only per role: per-role xp is collinear with gold and destabilizes attribution (drop it)
FEATURES = [c for c in panel.columns
            if c not in ("match_id", "winning_team", "game_duration", "label")
            and not c.startswith("xp_")]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(panel, groups=panel.match_id))
train, test = panel.iloc[tr_idx], panel.iloc[te_idx]

def mirror(df):
    m = df.copy()
    for c in FEATURES:
        if c != "minute":
            m[c] = -m[c]
    m["label"] = 1 - m["label"]
    return pd.concat([df, m], ignore_index=True)

train_m = mirror(train)
# monotonic prior: every lead (gold/objective) can only help, never hurt. minute unconstrained.
mono = [0 if f == "minute" else 1 for f in FEATURES]
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                     l2_regularization=1.0, monotonic_cst=mono, random_state=42)
clf.fit(train_m[FEATURES], train_m["label"])
joblib.dump({"model": clf, "features": FEATURES, "roles": ROLES}, "data/win_model2.joblib")

p = clf.predict_proba(test[FEATURES])[:, 1]
print(f"v2 OVERALL test AUC={roc_auc_score(test.label, p):.4f}  Brier={brier_score_loss(test.label, p):.4f}  (v1 was 0.805 / 0.180)")
print("\nAUC by minute:", end="  ")
for lo, hi in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
    seg = test[(test.minute >= lo) & (test.minute < hi)]
    print(f"{lo}-{hi}:{roc_auc_score(seg.label, clf.predict_proba(seg[FEATURES])[:,1]):.3f}", end="  ")

# --- role-specific marginal weights (AME) ---
X = panel[FEATURES].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]

def ame(feat, unit):
    Xp = X.copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100

def at_even(feat, minute, unit):
    row = {c: 0.0 for c in FEATURES}; row["minute"] = minute
    a = clf.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1]
    row[feat] += unit
    return (clf.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1] - a) * 100

print("\n\n== role-specific GOLD value (per 1000 gold lead on that role) ==")
print(f"{'role':<9}{'AME ΔP%':>10}{'even min20':>12}")
for r in ROLES:
    print(f"{r:<9}{ame('gold_'+r,1000):>+10.3f}{at_even('gold_'+r,20,1000):>+12.3f}")
print("\n== team objectives (per 1) ==")
for f in ["kill_diff","tower_diff","inhib_diff","plate_diff","dragon_diff","baron_diff","herald_diff","grub_diff"]:
    print(f"  {f:<12}{ame(f,1):>+9.3f}  (AME)   {at_even(f,20,1):>+8.3f}  (even min20)")
