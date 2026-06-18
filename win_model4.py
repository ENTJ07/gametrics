"""Win model v4 = v3 + CC (monotonic+) + damage-taken (unconstrained, to read its true sign).
Decides whether tanking-via-damage-taken is a usable channel."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import joblib

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
panel = pd.read_parquet("data/parsed/panel4.parquet")
panel = panel[panel.minute >= 1]
FEATURES = [c for c in panel.columns
            if c not in ("match_id", "winning_team", "game_duration", "label") and not c.startswith("xp_")]

gss = GroupShuffleSplit(1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(panel, groups=panel.match_id))
train, test = panel.iloc[tr_idx], panel.iloc[te_idx]

def mirror(df):
    m = df.copy()
    for c in FEATURES:
        if c != "minute":
            m[c] = -m[c]
    m["label"] = 1 - m["label"]
    return pd.concat([df, m], ignore_index=True)

# all leads monotonic+ (a lead can't hurt); minute unconstrained. dtaken+ tests if tanking survives the prior.
mono = [0 if f == "minute" else 1 for f in FEATURES]
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                     l2_regularization=1.0, monotonic_cst=mono, random_state=42)
clf.fit(mirror(train)[FEATURES], mirror(train)["label"])
joblib.dump({"model": clf, "features": FEATURES, "roles": ROLES}, "data/win_model4.joblib")
print(f"v4 AUC={roc_auc_score(test.label, clf.predict_proba(test[FEATURES])[:,1]):.4f}  (v3 was 0.799)")

X = panel[FEATURES].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]
def ame(feat, unit):
    Xp = X.copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100

gold_ref = np.mean([ame(f"gold_{r}", 1000) for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]])
cc_u = panel.cc_diff.abs().mean()       # natural unit ~ avg |cc lead|
dt_u = panel.dtaken_diff.abs().mean()
cc_w = ame("cc_diff", cc_u)
dt_w = ame("dtaken_diff", dt_u)
print(f"\nCC: per {cc_u:,.0f}u = {cc_w:+.3f}%  -> gold-equiv {cc_w/gold_ref*1000:+.0f}g   [USE: {'yes' if cc_w>0.001 else 'no'}]")
print(f"damage-taken: per {dt_u:,.0f}u = {dt_w:+.3f}%  -> {dt_w/gold_ref*1000:+.0f}g   [USE: {'yes' if dt_w>0.001 else 'NO (<=0, contaminated by losing)'}]")
print(f"\nvision still: {ame('vision_diff',1):+.4f}%/action  | role gold UTIL {ame('gold_UTILITY',1000):+.3f}")
