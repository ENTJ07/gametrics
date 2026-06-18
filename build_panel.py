"""Per-minute team-state panel for the win-expectancy model.
Row = (match_id, minute). Features = team100 - team200 diffs. Label = team100 wins.
Excludes remakes / early-surrender games."""
import numpy as np
import pandas as pd

meta = pd.read_parquet("data/parsed/match_meta.parquet")
good = meta[(~meta.is_remake) & (~meta.early_surrender)].copy()
good_ids = set(good.match_id)
print(f"clean games: {len(good):,} / {len(meta):,}")

# --- continuous state from participant frames ---
pf = pd.read_parquet("data/parsed/pframes.parquet",
                     columns=["match_id", "minute", "team_id", "total_gold", "xp", "minions", "jungle", "level"])
pf = pf[pf.match_id.isin(good_ids)]
pf["cs"] = pf.minions.fillna(0) + pf.jungle.fillna(0)
team = pf.groupby(["match_id", "minute", "team_id"]).agg(
    gold=("total_gold", "sum"), xp=("xp", "sum"), cs=("cs", "sum"), level=("level", "sum")).reset_index()
piv = team.pivot_table(index=["match_id", "minute"], columns="team_id", values=["gold", "xp", "cs", "level"])
panel = pd.DataFrame(index=piv.index)
for m in ["gold", "xp", "cs", "level"]:
    panel[m + "_diff"] = piv[(m, 100)] - piv[(m, 200)]
panel = panel.reset_index()
print(f"panel grid: {len(panel):,} (match,minute) rows")

# --- cumulative objectives from events ---
ev = pd.read_parquet("data/parsed/events.parquet")
ev = ev[ev.match_id.isin(good_ids)]

def add_cumdiff(panel, sub, gain_team, colname):
    """gain_team: Series of team (100/200) that GAINED the objective. Adds cumulative team100-team200 diff."""
    s = pd.DataFrame({"match_id": sub.match_id.values, "minute": sub.minute.values,
                      "signed": np.where(gain_team == 100, 1, -1)})
    inc = s.groupby(["match_id", "minute"]).signed.sum().reset_index(name="inc")
    panel = panel.merge(inc, on=["match_id", "minute"], how="left")
    panel["inc"] = panel["inc"].fillna(0)
    panel[colname] = panel.groupby("match_id")["inc"].cumsum()
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
# kills (cumulative) from CHAMPION_KILL: gaining team = team of killer
ck = ev[ev.type == "CHAMPION_KILL"].copy()
ck_gain = np.where(ck.killer_id <= 5, 100, 200)  # killer 1-5 = team100
panel = add_cumdiff(panel, ck, ck_gain, "kill_diff")

# --- label ---
panel = panel.merge(good[["match_id", "winning_team", "game_duration"]], on="match_id", how="left")
panel["label"] = (panel.winning_team == 100).astype(int)
panel = panel[panel.minute <= panel.game_duration / 60 + 1]  # drop stray post-game frames

panel.to_parquet("data/parsed/panel.parquet", index=False)
print(f"\nwrote panel: {len(panel):,} rows x {panel.shape[1]} cols")

# --- verify objective sign conventions: winners should lead in objectives ---
print("\n== sign check: mean diff at minute 20, grouped by team100 win/loss ==")
m20 = panel[panel.minute == 20]
chk = m20.groupby("label")[["gold_diff", "tower_diff", "dragon_diff", "kill_diff", "plate_diff"]].mean()
chk.index = ["team100 LOST", "team100 WON"]
print(chk.round(2).to_string())
print("\n(WON row should be positive across the board if conventions are correct)")
