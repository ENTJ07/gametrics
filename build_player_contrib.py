"""Per-player per-game RAW contribution quantities, by channel.
Channels valued (vs role replacement) later in war.py. Credit-split for shared events here."""
import numpy as np
import pandas as pd

meta = pd.read_parquet("data/parsed/match_meta.parquet")
good_ids = set(meta[(~meta.is_remake) & (~meta.early_surrender)].match_id)

part = pd.read_parquet("data/parsed/participants.parquet")
part = part[part.match_id.isin(good_ids)].copy()
base = part[["match_id", "participant_id", "puuid", "team_position", "team_id", "win", "champion",
             "gold_earned", "cs", "vision_score", "wards_placed", "control_wards", "wards_killed",
             "time_cc_others", "dmg_champ", "dmg_taken", "dmg_mitigated", "heal_teammates",
             "shield_teammates", "kills", "assists", "deaths"]].copy()
base = base.rename(columns={"assists": "assists_n"})

ev = pd.read_parquet("data/parsed/events.parquet")
ev = ev[ev.match_id.isin(good_ids)].copy()

def split_credit(sub, mode):
    """Distribute 1.0 of each event among involved players. mode 'kill' = killer-weighted, else equal."""
    rows = []
    for mid, killer, assists in zip(sub.match_id.values, sub.killer_id.values, sub.assists.values):
        al = [int(a) for a in assists.split(",")] if isinstance(assists, str) and assists else []
        k = int(killer) if not pd.isna(killer) else 0
        if mode == "kill":
            inv = ([(k, 0.6)] + [(a, 0.4 / len(al)) for a in al]) if al else [(k, 1.0)]
        else:
            ps = ([k] if k >= 1 else []) + al
            inv = [(p, 1.0 / len(ps)) for p in ps] if ps else []
        for p, c in inv:
            if p >= 1:
                rows.append((mid, p, c))
    return pd.DataFrame(rows, columns=["match_id", "participant_id", "credit"])

def credit_col(sub, mode, name):
    cr = split_credit(sub, mode)
    if cr.empty:
        return None
    g = cr.groupby(["match_id", "participant_id"]).credit.sum().reset_index(name=name)
    return g

specs = [
    (ev[ev.type == "CHAMPION_KILL"], "kill", "kill_credit"),
    (ev[(ev.type == "ELITE_MONSTER_KILL") & (ev.monster_type == "DRAGON")], "eq", "dragon_credit"),
    (ev[(ev.type == "ELITE_MONSTER_KILL") & (ev.monster_type == "BARON_NASHOR")], "eq", "baron_credit"),
    (ev[(ev.type == "ELITE_MONSTER_KILL") & (ev.monster_type == "RIFTHERALD")], "eq", "herald_credit"),
    (ev[(ev.type == "ELITE_MONSTER_KILL") & (ev.monster_type == "HORDE")], "eq", "grub_credit"),
    (ev[(ev.type == "BUILDING_KILL") & (ev.building_type == "TOWER_BUILDING")], "eq", "tower_credit"),
    (ev[(ev.type == "BUILDING_KILL") & (ev.building_type == "INHIBITOR_BUILDING")], "eq", "inhib_credit"),
    (ev[ev.type == "TURRET_PLATE_DESTROYED"], "eq", "plate_credit"),
]
out = base
for sub, mode, name in specs:
    g = credit_col(sub, mode, name)
    if g is not None:
        out = out.merge(g, on=["match_id", "participant_id"], how="left")
    out[name] = out.get(name, 0)
    out[name] = out[name].fillna(0)

out.to_parquet("data/parsed/player_contrib.parquet", index=False)
print(f"wrote player_contrib: {len(out):,} rows x {out.shape[1]} cols")

# --- sanity: do winning-team players accumulate more credit? by role ---
print("\n== mean credited events by role x win (should: winners > losers everywhere) ==")
chk = out.groupby(["team_position", "win"])[["kill_credit", "dragon_credit", "tower_credit", "gold_earned"]].mean().round(1)
print(chk.to_string())
print("\n== role gold profile (why support gold-channel is ~0) ==")
print(out.groupby("team_position")[["gold_earned", "cs", "vision_score", "time_cc_others", "kill_credit"]].mean().round(0).to_string())
