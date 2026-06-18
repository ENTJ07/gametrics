"""Which PER-MINUTE metrics are repeatable (skill) vs noise, by role?
Per-minute normalization removes game-length noise. Low K = real individual skill signal."""
import numpy as np
import pandas as pd

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
pc = pd.read_parquet("data/parsed/player_contrib.parquet")
meta = pd.read_parquet("data/parsed/match_meta.parquet")[["match_id", "game_duration"]]
pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "participant_id", "time_enemy_controlled", "dmg_taken"])
pcc = pf.groupby(["match_id", "participant_id"]).agg(player_cc=("time_enemy_controlled", "max"),
                                                     player_dtaken=("dmg_taken", "max")).reset_index()
pc = pc.merge(pcc, on=["match_id", "participant_id"], how="left").merge(meta, on="match_id", how="left")
mins = pc.game_duration / 60.0

RAW = ["gold_earned", "cs", "kill_credit", "assists_n", "vision_score", "wards_placed",
       "control_wards", "wards_killed", "player_cc", "player_dtaken", "dmg_champ", "deaths"]
for m in RAW:
    pc[m + "_pm"] = pc[m] / mins
METRICS = [m + "_pm" for m in RAW]

def K_of(sub, metric):
    g = sub.dropna(subset=[metric])
    by = g.groupby("puuid")[metric]
    cnt = by.size()
    wv = by.var()[cnt >= 10]
    if len(wv) < 20:
        return np.nan
    sigma2 = wv.mean()
    means = by.mean()
    var_true = max(means.var() - sigma2 * (1 / cnt).mean(), 1e-12)
    return sigma2 / var_true

print("K = games to half-stabilize (LOW=repeatable skill). per-minute metrics, by role:\n")
print(f"{'metric (per min)':<18}" + "".join(f"{r[:4]:>7}" for r in ROLES))
rows = {}
for m in METRICS:
    rows[m] = {r: K_of(pc[pc.team_position == r], m) for r in ROLES}
    line = f"{m[:-3]:<18}"
    for r in ROLES:
        k = rows[m][r]
        line += f"{min(k,99999):>7.0f}" if not np.isnan(k) else f"{'-':>7}"
    print(line)

print("\n=> UTILITY most-repeatable (lowest K) = usable support-skill basis:")
util = sorted([(m, rows[m]["UTILITY"]) for m in METRICS if not np.isnan(rows[m]["UTILITY"])], key=lambda t: t[1])
for m, k in util[:6]:
    print(f"   {m[:-3]:<18} K={k:>5.0f}  reliability@30g={30/(30+k):.2f}")
print("\n=> for contrast, BOTTOM(adc) most-repeatable:")
bot = sorted([(m, rows[m]["BOTTOM"]) for m in METRICS if not np.isnan(rows[m]["BOTTOM"])], key=lambda t: t[1])
for m, k in bot[:4]:
    print(f"   {m[:-3]:<18} K={k:>5.0f}")
