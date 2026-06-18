"""Win model v3 = v2 (role gold + objectives) + leak-free vision state. Monotonic.
Decides the gold-equivalent weight of the vision channel."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
panel = pd.read_parquet("data/parsed/panel3.parquet")
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

train_m = mirror(train)
mono = [0 if f == "minute" else 1 for f in FEATURES]
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=6,
                                     l2_regularization=1.0, monotonic_cst=mono, random_state=42)
clf.fit(train_m[FEATURES], train_m["label"])
joblib.dump({"model": clf, "features": FEATURES, "roles": ROLES}, "data/win_model3.joblib")

p = clf.predict_proba(test[FEATURES])[:, 1]
print(f"v3 AUC={roc_auc_score(test.label, p):.4f}  Brier={brier_score_loss(test.label, p):.4f}  (v2 was 0.799)")

X = panel[FEATURES].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]
def ame(feat, unit):
    Xp = X.copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100

print("\nrole gold (%/1000g):", {r: round(ame(f"gold_{r}", 1000), 2) for r in ROLES})
gold_ref = np.mean([ame(f"gold_{r}", 1000) for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]])
v1 = ame("vision_diff", 1)
v10 = ame("vision_diff", 10)
print(f"\nVISION weight: per 1 action {v1:+.4f}%  | per 10 {v10:+.3f}%")
print(f"  gold-equivalent: 1 vision action = {v1/gold_ref*1000:+.0f} gold  (carry-gold ref {gold_ref:.2f}%/1000g)")
print(f"\nkey question answered: vision net-of-gold value is {'POSITIVE -> supports get credit' if v1 > 0.001 else 'about zero'}")
