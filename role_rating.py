"""Within-role SKILL rating (0-100) on each role's REPEATABLE per-minute metrics,
oriented by within-role win-correlation. Validated by split-half reliability.
Solves the support problem: rate each role on its own craft, not shared win-contribution."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
rng = np.random.RandomState(7)

pc = pd.read_parquet("data/parsed/player_contrib.parquet")
meta = pd.read_parquet("data/parsed/match_meta.parquet")[["match_id", "game_duration"]]
pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "participant_id", "time_enemy_controlled", "dmg_taken"])
pcc = pf.groupby(["match_id", "participant_id"]).agg(player_cc=("time_enemy_controlled", "max"),
                                                     player_dtaken=("dmg_taken", "max")).reset_index()
pc = pc.merge(pcc, on=["match_id", "participant_id"], how="left").merge(meta, on="match_id", how="left")
mins = (pc.game_duration / 60.0).clip(lower=1)
# repeatable per-minute craft metrics (K<=~15 for most roles); deaths included (model signs it)
FEATS = ["gold_earned", "cs", "dmg_champ", "vision_score", "control_wards", "wards_placed",
         "wards_killed", "player_cc", "player_dtaken", "kill_credit", "deaths"]
for m in FEATS:
    pc[m + "_pm"] = pc[m] / mins
F = [m + "_pm" for m in FEATS]
pc["half"] = rng.randint(0, 2, len(pc))

out = []
print(f"{'role':<8}{'n_games':>8}{'win-AUC':>9}{'split-half r':>14}  top within-role skill drivers")
for role in ROLES:
    d = pc[pc.team_position == role].dropna(subset=F).copy()
    sc = StandardScaler().fit(d[F])
    Z = sc.transform(d[F])
    lr = LogisticRegression(max_iter=1000, C=0.5).fit(Z, d.win.astype(int))
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(d.win, lr.predict_proba(Z)[:, 1])
    w = lr.coef_[0]
    d["idx"] = Z @ w

    # player skill index = mean of per-game idx; split-half reliability
    pl = d.groupby("puuid").agg(idx=("idx", "mean"), games=("idx", "size"), win=("win", "mean"))
    ha = d[d.half == 0].groupby("puuid").idx.mean()
    hb = d[d.half == 1].groupby("puuid").idx.mean()
    cga = d[d.half == 0].groupby("puuid").size(); cgb = d[d.half == 1].groupby("puuid").size()
    ok = (cga >= 7) & (cgb >= 7)
    shr = ha[ok[ok].index].corr(hb[ok[ok].index])

    drivers = sorted(zip(FEATS, w), key=lambda t: -abs(t[1]))[:4]
    drv = ", ".join(f"{n}{'+' if c>0 else '-'}" for n, c in drivers)
    print(f"{role:<8}{len(d):>8}{auc:>9.3f}{shr:>14.3f}  {drv}")

    pl["role"] = role
    pl["skill"] = pl.idx.rank(pct=True) * 100
    pl["winrate"] = pl.win
    out.append(pl.reset_index())

res = pd.concat(out, ignore_index=True)
res.to_parquet("data/parsed/role_skill.parquet", index=False)

print("\n=== TOP 8 SUPPORTS by skill rating (finally rateable) ===")
sup = res[(res.role == "UTILITY") & (res.games >= 15)].sort_values("skill", ascending=False).head(8)
print(f"{'skill':>6}{'games':>7}{'winrate':>9}")
for _, r in sup.iterrows():
    print(f"{r.skill:>6.0f}{r.games:>7.0f}{r.winrate:>8.0%}")
print(f"\nsupport skill vs winrate corr: {res[(res.role=='UTILITY')&(res.games>=15)].skill.corr(res[(res.role=='UTILITY')&(res.games>=15)].winrate):.3f}")
