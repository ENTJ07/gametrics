"""Freeze all trained components into data/artifacts.joblib so a fresh player can be scored
on-demand from raw match JSON, consistently with the precomputed dataset."""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_LABEL = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "ADC", "UTILITY": "Support"}

# ---- 1. channel weights (AME on win model v4) ----
d = joblib.load("data/win_model4.joblib")
clf, F = d["model"], d["features"]
panel = pd.read_parquet("data/parsed/panel4.parquet")
panel = panel[panel.minute >= 1]
X = panel[F].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]
def ame(feat, unit):
    Xp = X.copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100
role_gold_w = {r: ame(f"gold_{r}", 1000) / 1000 for r in ROLES}      # per 1 gold
STRUCT = {"kill_credit": "kill_diff", "dragon_credit": "dragon_diff", "baron_credit": "baron_diff",
          "herald_credit": "herald_diff", "grub_credit": "grub_diff", "tower_credit": "tower_diff",
          "inhib_credit": "inhib_diff", "plate_credit": "plate_diff"}
struct_w = {c: ame(f, 1) for c, f in STRUCT.items()}
vision_w, cc_w, tank_w = ame("vision_diff", 1), ame("cc_diff", 1), ame("dtaken_diff", 1)

# ---- 2. role baselines (channel quantity means) ----
pc = pd.read_parquet("data/parsed/player_contrib.parquet")
pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "participant_id", "time_enemy_controlled", "dmg_taken"])
pcc = pf.groupby(["match_id", "participant_id"]).agg(player_cc=("time_enemy_controlled", "max"),
                                                     player_dtaken=("dmg_taken", "max")).reset_index()
pc = pc.merge(pcc, on=["match_id", "participant_id"], how="left")
pc["vision_actions"] = pc.wards_placed.fillna(0) + pc.wards_killed.fillna(0)
QTY = ["gold_earned", "vision_actions", "player_cc", "player_dtaken"] + list(STRUCT)
baselines = pc.groupby("team_position")[QTY].mean().to_dict("index")

# ---- 3. per-role skill model (per-minute craft) ----
meta = pd.read_parquet("data/parsed/match_meta.parquet")[["match_id", "game_duration"]]
pc = pc.merge(meta, on="match_id", how="left")
mins = (pc.game_duration / 60.0).clip(lower=1)
SK_FEATS = ["gold_earned", "cs", "dmg_champ", "vision_score", "control_wards", "wards_placed",
            "wards_killed", "player_cc", "player_dtaken", "kill_credit", "deaths"]
for m in SK_FEATS:
    pc[m + "_pm"] = pc[m] / mins
Fpm = [m + "_pm" for m in SK_FEATS]
skill_models = {}
for role in ROLES:
    dr = pc[pc.team_position == role].dropna(subset=Fpm).copy()
    sc = StandardScaler().fit(dr[Fpm])
    lr = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(dr[Fpm]), dr.win.astype(int))
    dr["idx"] = sc.transform(dr[Fpm]) @ lr.coef_[0]
    ref_idx = np.sort(dr.groupby("puuid").idx.mean().values)   # player-level reference for percentile
    skill_models[role] = {"feats": SK_FEATS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                          "coef": lr.coef_[0].tolist(), "ref_idx": ref_idx}

# ---- 4. replacement rates + reference distributions ----
war = pd.read_parquet("data/parsed/player_war.parquet")
P = 10
repl_rate = {r: float(np.percentile(war[(war.role == r) & (war.games >= 20)].waa_per, P)) for r in ROLES}
ref_war = {r: np.sort(war[(war.role == r) & (war.games >= 15)].war.values) for r in ROLES}
ref_warall = np.sort(war[war.games >= 15].war.values)

art = {"role_gold_w": role_gold_w, "struct_w": struct_w, "vision_w": vision_w, "cc_w": cc_w, "tank_w": tank_w,
       "struct_map": STRUCT, "baselines": baselines, "repl_rate": repl_rate,
       "skill_models": skill_models, "ref_war": ref_war, "ref_warall": ref_warall,
       "role_label": ROLE_LABEL, "patch": "16.12"}
joblib.dump(art, "data/artifacts.joblib")
print("saved data/artifacts.joblib")
print("role gold/1k:", {r: round(v*1000, 2) for r, v in role_gold_w.items()})
print("struct:", {k: round(v, 2) for k, v in struct_w.items()})
print("vision/cc/tank weights:", round(vision_w, 4), round(cc_w, 6), round(tank_w, 6))
print("repl_rate:", {r: round(v, 4) for r, v in repl_rate.items()})
