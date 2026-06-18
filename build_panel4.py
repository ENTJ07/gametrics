"""Add leak-free CC + damage-taken state from participant frames (both already cumulative).
cc_diff = team timeEnemySpentControlled diff; dtaken_diff = team totalDamageTaken diff."""
import numpy as np
import pandas as pd

panel = pd.read_parquet("data/parsed/panel3.parquet")
good_ids = set(panel.match_id.unique())
pf = pd.read_parquet("data/parsed/pframes.parquet",
                     columns=["match_id", "minute", "participant_id", "team_id", "time_enemy_controlled", "dmg_taken"])
pf = pf[pf.match_id.isin(good_ids)]

# verify cumulative (monotonic per participant)
chk = pf.sort_values(["match_id", "participant_id", "minute"]).groupby(["match_id", "participant_id"]).time_enemy_controlled
mono_frac = chk.apply(lambda s: (s.diff().dropna() >= -1).mean()).mean()
print(f"timeEnemySpentControlled monotonic fraction: {mono_frac:.3f} (~1 => cumulative)")

team = pf.groupby(["match_id", "minute", "team_id"]).agg(cc=("time_enemy_controlled", "sum"),
                                                         dtk=("dmg_taken", "sum")).reset_index()
piv = team.pivot_table(index=["match_id", "minute"], columns="team_id", values=["cc", "dtk"])
add = pd.DataFrame(index=piv.index)
add["cc_diff"] = piv[("cc", 100)] - piv[("cc", 200)]
add["dtaken_diff"] = piv[("dtk", 100)] - piv[("dtk", 200)]
add = add.reset_index()
panel = panel.merge(add, on=["match_id", "minute"], how="left")
panel[["cc_diff", "dtaken_diff"]] = panel[["cc_diff", "dtaken_diff"]].fillna(0)
panel.to_parquet("data/parsed/panel4.parquet", index=False)
print(f"wrote panel4: {len(panel):,} rows")

m20 = panel[panel.minute == 20]
print(f"\ncc_diff@20: mean|.|={m20.cc_diff.abs().mean():.0f}  by win: {m20.groupby('label').cc_diff.mean().round(0).to_dict()}")
print(f"dtaken_diff@20: mean|.|={m20.dtaken_diff.abs().mean():.0f}  by win: {m20.groupby('label').dtaken_diff.mean().round(0).to_dict()}")
