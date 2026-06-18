"""Do vision / CC / peel carry win-value BEYOND gold+kills+objectives?
Game-level team-diff model, monotonic. Decides whether support's non-gold channels get real weight."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

pc = pd.read_parquet("data/parsed/player_contrib.parquet")
pc["peel"] = pc.heal_teammates.fillna(0) + pc.shield_teammates.fillna(0)
STATS = ["gold_earned", "kill_credit", "dragon_credit", "baron_credit", "herald_credit", "grub_credit",
         "tower_credit", "inhib_credit", "plate_credit", "vision_score", "control_wards", "wards_killed",
         "time_cc_others", "dmg_mitigated", "peel", "dmg_champ"]
team = pc.groupby(["match_id", "team_id"]).agg({s: "sum" for s in STATS} | {"win": "first"}).reset_index()
p100 = team[team.team_id == 100].set_index("match_id")
p200 = team[team.team_id == 200].set_index("match_id")
diff = (p100[STATS] - p200[STATS]).dropna()
y = p100.loc[diff.index, "win"].astype(int)

gss = GroupShuffleSplit(1, test_size=0.2, random_state=1)
tr, te = next(gss.split(diff, groups=diff.index))
Xtr = pd.concat([diff.iloc[tr], -diff.iloc[tr]]); ytr = pd.concat([y.iloc[tr], 1 - y.iloc[tr]])
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                     monotonic_cst=[1] * len(STATS), random_state=1)
clf.fit(Xtr[STATS], ytr)
auc = roc_auc_score(y.iloc[te], clf.predict_proba(diff.iloc[te][STATS])[:, 1])
print(f"game-level team model AUC={auc:.3f}  n={len(diff):,}")

base = clf.predict_proba(diff[STATS])[:, 1]
def ame(feat, unit):
    Xp = diff[STATS].copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100

g_per1000 = ame("gold_earned", 1000)
print(f"\ngold AME per 1000 = {g_per1000:+.3f}%  -> gold-equivalent converter\n")
UNITS = {"gold_earned": 1000, "kill_credit": 1, "vision_score": 10, "control_wards": 1, "wards_killed": 5,
         "time_cc_others": 10, "dmg_mitigated": 1000, "peel": 1000, "dmg_champ": 1000,
         "dragon_credit": 1, "baron_credit": 1, "tower_credit": 1}
print(f"{'stat':<16}{'unit':>7}{'AME ΔP%':>10}{'gold-equiv':>12}")
for s, u in UNITS.items():
    a = ame(s, u)
    print(f"{s:<16}{u:>7}{a:>+10.3f}{a/g_per1000*1000:>+12.0f}")
