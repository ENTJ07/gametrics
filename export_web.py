"""Bake backend outputs into web/data.json WITH per-role and per-champion breakdowns,
using the same score_games aggregator as live lookups (single source of truth)."""
import os, json
import numpy as np
import pandas as pd
from score import score_games

pc = pd.read_parquet("data/parsed/player_contrib.parquet")
pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "participant_id", "time_enemy_controlled", "dmg_taken"])
pcc = pf.groupby(["match_id", "participant_id"]).agg(player_cc=("time_enemy_controlled", "max"),
                                                     player_dtaken=("dmg_taken", "max")).reset_index()
meta = pd.read_parquet("data/parsed/match_meta.parquet")[["match_id", "game_duration"]]
names = pd.read_parquet("data/parsed/names.parquet").set_index("puuid")
pc = pc.merge(pcc, on=["match_id", "participant_id"], how="left").merge(meta, on="match_id", how="left")

NUM = ["gold_earned", "cs", "dmg_champ", "vision_score", "control_wards", "wards_placed", "wards_killed",
       "deaths", "kill_credit", "dragon_credit", "baron_credit", "herald_credit", "grub_credit",
       "tower_credit", "inhib_credit", "plate_credit", "player_cc", "player_dtaken"]
pc[NUM] = pc[NUM].fillna(0)

by = {}
for r in pc.itertuples():
    g = {"role": r.team_position, "win": bool(r.win), "minutes": max(r.game_duration / 60.0, 1),
         "champion": r.champion,
         "q": {"gold_earned": r.gold_earned, "vision_actions": r.wards_placed + r.wards_killed,
               "kill_credit": r.kill_credit, "dragon_credit": r.dragon_credit, "baron_credit": r.baron_credit,
               "herald_credit": r.herald_credit, "grub_credit": r.grub_credit, "tower_credit": r.tower_credit,
               "inhib_credit": r.inhib_credit, "plate_credit": r.plate_credit,
               "player_cc": r.player_cc, "player_dtaken": r.player_dtaken},
         "craft": {"gold_earned": r.gold_earned, "cs": r.cs, "dmg_champ": r.dmg_champ,
                   "vision_score": r.vision_score, "control_wards": r.control_wards, "wards_placed": r.wards_placed,
                   "wards_killed": r.wards_killed, "kill_credit": r.kill_credit, "deaths": r.deaths,
                   "player_cc": r.player_cc, "player_dtaken": r.player_dtaken}}
    by.setdefault(r.puuid, []).append(g)

records = []
for pu, gs in by.items():
    if len(gs) < 10:
        continue
    rec = score_games(gs)
    if not rec:
        continue
    nm = names.loc[pu] if pu in names.index else None
    rec["name"] = (str(nm["name"]).strip() if nm is not None else "") or "Unknown"
    rec["tag"] = (str(nm["tag"]).strip() if nm is not None else "")
    rec["champions"] = rec["champions"][:8]
    records.append(rec)

records.sort(key=lambda x: -x["war"])
wars = np.sort([r["war"] for r in records])
for r in records:
    r["war_rank"] = int(len(wars) - np.searchsorted(wars, r["war"]))
# skill rank within main role
for role in set(r["role"] for r in records):
    pool = sorted([r for r in records if r["role"] == role and r["skill"] is not None], key=lambda x: -x["skill"])
    for i, r in enumerate(pool):
        r["skill_rank"] = i + 1

rc = {}
for r in records:
    rc[r["role"]] = rc.get(r["role"], 0) + 1
data = {"meta": {"patch": "16.12", "matches": 10000, "players": len(records),
                 "region": "KR Challenger / GM / Master", "roles": rc}, "players": records}
os.makedirs("web", exist_ok=True)
json.dump(data, open("web/data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"wrote web/data.json: {len(records):,} players ({os.path.getsize('web/data.json')/1024:.0f} KB)")
# show a multi-role example
ex = max(records, key=lambda r: len(r["roles"]))
print(f"\nmulti-role example: {ex['name']}#{ex['tag']} ({len(ex['roles'])} roles, {len(ex['champions'])} champs)")
for rr in ex["roles"]:
    print(f"  {rr['role']:<8} {rr['games']:>3}g  WAR {rr['war']:>5}  skill {rr['skill']}  WR {rr['winrate']}%")
for c in ex["champions"][:4]:
    print(f"   {c['champion']:<12} {c['games']:>3}g  WAR {c['war']:>5}  WR {c['winrate']}%  ({c['role']})")
