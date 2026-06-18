"""Wins Above Average, gold-equivalent currency, 6 leak-free channels:
economy(role gold), combat(kills), objectives, vision, CC, tanking(dmg absorbed).
Each channel = (player - role mean) x leak-free time-resolved weight."""
import numpy as np
import pandas as pd
import joblib

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
d = joblib.load("data/win_model4.joblib")
clf, F = d["model"], d["features"]
panel = pd.read_parquet("data/parsed/panel4.parquet")
panel = panel[panel.minute >= 1]
X = panel[F].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]

def ame(feat, unit):
    Xp = X.copy(); Xp[feat] += unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100

role_gold_w = {r: ame(f"gold_{r}", 1000) for r in ROLES}
STRUCT = {"kill_credit": "kill_diff", "dragon_credit": "dragon_diff", "baron_credit": "baron_diff",
          "herald_credit": "herald_diff", "grub_credit": "grub_diff", "tower_credit": "tower_diff",
          "inhib_credit": "inhib_diff", "plate_credit": "plate_diff"}
struct_w = {c: ame(f, 1) for c, f in STRUCT.items()}
vision_w, cc_w, tank_w = ame("vision_diff", 1), ame("cc_diff", 1), ame("dtaken_diff", 1)

# per-player final cumulative CC and damage-taken (timeline-only fields)
pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "participant_id", "time_enemy_controlled", "dmg_taken"])
pcc = pf.groupby(["match_id", "participant_id"]).agg(player_cc=("time_enemy_controlled", "max"),
                                                     player_dtaken=("dmg_taken", "max")).reset_index()

pc = pd.read_parquet("data/parsed/player_contrib.parquet").merge(pcc, on=["match_id", "participant_id"], how="left")
pc["vision_actions"] = pc.wards_placed.fillna(0) + pc.wards_killed.fillna(0)
QTY = ["gold_earned", "vision_actions", "player_cc", "player_dtaken"] + list(STRUCT)
rm = pc.groupby("team_position")[QTY].mean().rename(columns=lambda c: c + "_rm")
pc = pc.merge(rm, left_on="team_position", right_index=True, how="left")

pc["econ_wp"] = (pc.gold_earned - pc.gold_earned_rm) / 1000 * pc.team_position.map(role_gold_w)
pc["combat_wp"] = (pc.kill_credit - pc.kill_credit_rm) * struct_w["kill_credit"]
pc["obj_wp"] = sum((pc[c] - pc[c + "_rm"]) * struct_w[c] for c in STRUCT if c != "kill_credit")
pc["vision_wp"] = (pc.vision_actions - pc.vision_actions_rm) * vision_w
pc["cc_wp"] = (pc.player_cc - pc.player_cc_rm) * cc_w
pc["tank_wp"] = (pc.player_dtaken - pc.player_dtaken_rm) * tank_w
CH = ["econ_wp", "combat_wp", "obj_wp", "vision_wp", "cc_wp", "tank_wp"]
pc["waa_game"] = pc[CH].sum(1) / 100.0
pc[["puuid", "match_id", "team_position", "waa_game", "win"]].to_parquet("data/parsed/player_game_waa.parquet", index=False)
pc.groupby("puuid")[CH].sum().to_parquet("data/parsed/player_channels.parquet")

print(f"WAA mean | winners {pc[pc.win].waa_game.mean():+.4f}  losers {pc[~pc.win].waa_game.mean():+.4f}")
print("\nchannel share of |WAA| by role (6 channels):")
chan = pc.groupby("team_position")[CH].agg(lambda s: s.abs().mean())
share = (chan.div(chan.sum(1), 0) * 100).round(0).astype(int)
share.columns = ["econ", "combat", "obj", "vision", "cc", "tank"]
print(share.to_string())

agg = pc.groupby("puuid").agg(games=("waa_game", "size"), waa=("waa_game", "sum"),
        waa_per=("waa_game", "mean"), role=("team_position", lambda s: s.mode().iloc[0]),
        winrate=("win", "mean")).reset_index()
agg.to_parquet("data/parsed/player_waa.parquet", index=False)
elig = agg[agg.games >= 15].copy()
print(f"\n=== TOP 12 overall (>=15 games, {len(elig):,} players) ===")
print(f"{'role':<8}{'games':>6}{'WAA':>7}{'WAA/g':>8}{'winrate':>9}")
for _, r in elig.sort_values("waa", ascending=False).head(12).iterrows():
    print(f"{r.role:<8}{r.games:>6.0f}{r.waa:>7.2f}{r.waa_per:>8.3f}{r.winrate:>8.0%}")
print("\nbest player per role (total WAA):")
for role in ROLES:
    rr = elig[elig.role == role].sort_values("waa", ascending=False).head(1)
    if len(rr):
        x = rr.iloc[0]
        rank = (elig.waa > x.waa).sum() + 1
        print(f"  {role:<8} WAA {x.waa:>6.2f}  WAA/g {x.waa_per:+.3f}  winrate {x.winrate:.0%}  (overall #{rank})")
print(f"\nWAA/game vs winrate corr: {elig.waa_per.corr(elig.winrate):.3f}")
