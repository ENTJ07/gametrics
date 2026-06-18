"""Role-resolved per-minute panel for win model v2.
Resources (gold, xp) split by role (team100 - team200 within each role). Objectives stay team-level."""
import numpy as np
import pandas as pd

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

meta = pd.read_parquet("data/parsed/match_meta.parquet")
good = meta[(~meta.is_remake) & (~meta.early_surrender)].copy()
good_ids = set(good.match_id)
print(f"clean games: {len(good):,}")

part = pd.read_parquet("data/parsed/participants.parquet", columns=["match_id", "participant_id", "team_position"])
pf = pd.read_parquet("data/parsed/pframes.parquet",
                     columns=["match_id", "minute", "participant_id", "team_id", "total_gold", "xp"])
pf = pf[pf.match_id.isin(good_ids)].merge(part, on=["match_id", "participant_id"], how="left")
pf = pf[pf.team_position.isin(ROLES)]

agg = pf.groupby(["match_id", "minute", "team_position", "team_id"]).agg(
    gold=("total_gold", "sum"), xp=("xp", "sum")).reset_index()
piv = agg.pivot_table(index=["match_id", "minute"], columns=["team_position", "team_id"], values=["gold", "xp"])
panel = pd.DataFrame(index=piv.index)
for r in ROLES:
    for met in ["gold", "xp"]:
        c1, c2 = (met, r, 100), (met, r, 200)
        if c1 in piv.columns and c2 in piv.columns:
            panel[f"{met}_{r}"] = piv[c1] - piv[c2]
panel = panel.fillna(0.0).reset_index()
print(f"panel grid: {len(panel):,} rows, role-resource cols: {panel.shape[1]-2}")

# --- team-level objectives (reuse cumulative-diff logic) ---
ev = pd.read_parquet("data/parsed/events.parquet")
ev = ev[ev.match_id.isin(good_ids)]

def add_cumdiff(panel, sub, gain_team, col):
    s = pd.DataFrame({"match_id": sub.match_id.values, "minute": sub.minute.values,
                      "signed": np.where(gain_team == 100, 1, -1)})
    inc = s.groupby(["match_id", "minute"]).signed.sum().reset_index(name="inc")
    panel = panel.merge(inc, on=["match_id", "minute"], how="left")
    panel["inc"] = panel["inc"].fillna(0)
    panel[col] = panel.groupby("match_id")["inc"].cumsum()
    return panel.drop(columns="inc")

tower = ev[(ev.type == "BUILDING_KILL") & (ev.building_type == "TOWER_BUILDING")]
panel = add_cumdiff(panel, tower, np.where(tower.team_id == 100, 200, 100), "tower_diff")
inhib = ev[(ev.type == "BUILDING_KILL") & (ev.building_type == "INHIBITOR_BUILDING")]
panel = add_cumdiff(panel, inhib, np.where(inhib.team_id == 100, 200, 100), "inhib_diff")
plate = ev[ev.type == "TURRET_PLATE_DESTROYED"]
panel = add_cumdiff(panel, plate, np.where(plate.team_id == 100, 200, 100), "plate_diff")
for mon, col in [("DRAGON", "dragon_diff"), ("BARON_NASHOR", "baron_diff"),
                 ("RIFTHERALD", "herald_diff"), ("HORDE", "grub_diff")]:
    sub = ev[(ev.type == "ELITE_MONSTER_KILL") & (ev.monster_type == mon)]
    panel = add_cumdiff(panel, sub, sub.team_id.values, col)
ck = ev[ev.type == "CHAMPION_KILL"]
panel = add_cumdiff(panel, ck, np.where(ck.killer_id <= 5, 100, 200), "kill_diff")

panel = panel.merge(good[["match_id", "winning_team", "game_duration"]], on="match_id", how="left")
panel["label"] = (panel.winning_team == 100).astype(int)
panel = panel[panel.minute <= panel.game_duration / 60 + 1]
panel.to_parquet("data/parsed/panel2.parquet", index=False)
print(f"wrote panel2: {len(panel):,} rows x {panel.shape[1]} cols")
print("cols:", [c for c in panel.columns if c not in ('match_id','winning_team','game_duration','label')])
