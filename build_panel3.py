"""Add leak-free time-resolved vision state to the panel.
vision_diff = cumulative (wards placed + enemy wards killed), team100 - team200, per minute."""
import numpy as np
import pandas as pd

panel = pd.read_parquet("data/parsed/panel2.parquet").sort_values(["match_id", "minute"])
good_ids = set(panel.match_id.unique())
ev = pd.read_parquet("data/parsed/events.parquet")
ev = ev[ev.match_id.isin(good_ids)]

wp = ev[ev.type == "WARD_PLACED"][["match_id", "minute", "creator_id"]].rename(columns={"creator_id": "actor"})
wk = ev[ev.type == "WARD_KILL"][["match_id", "minute", "killer_id"]].rename(columns={"killer_id": "actor"})
vis = pd.concat([wp, wk], ignore_index=True)
vis = vis[vis.actor >= 1]
vis["signed"] = np.where(vis.actor <= 5, 1, -1)
inc = vis.groupby(["match_id", "minute"]).signed.sum().reset_index(name="inc")

panel = panel.merge(inc, on=["match_id", "minute"], how="left")
panel["inc"] = panel["inc"].fillna(0)
panel["vision_diff"] = panel.groupby("match_id")["inc"].cumsum()
panel = panel.drop(columns="inc")
panel.to_parquet("data/parsed/panel3.parquet", index=False)

print(f"wrote panel3: {len(panel):,} rows")
m20 = panel[panel.minute == 20]
print(f"vision_diff at min20: mean|.|={m20.vision_diff.abs().mean():.1f}  range [{m20.vision_diff.min():.0f},{m20.vision_diff.max():.0f}]")
print("sign check (mean vision_diff by team100 win/loss at min20):")
print(m20.assign(won=m20.label).groupby("won").vision_diff.mean().round(2).to_string())
