"""Context-neutral linear weights: average marginal ΔP(win) per unit of each state feature.
Two views: (1) AME over the full state distribution (context-neutral weight),
(2) value at an even game by minute (interpretable headline)."""
import numpy as np
import pandas as pd
import joblib

d = joblib.load("data/win_model.joblib")
clf, F = d["model"], d["features"]

panel = pd.read_parquet("data/parsed/panel.parquet")
panel = panel[panel.minute >= 1]
X = panel[F].astype(float).reset_index(drop=True)
base = clf.predict_proba(X)[:, 1]

# unit per feature (structural objectives = 1; resources in natural chunks)
UNITS = {
    "kill_diff": 1, "dragon_diff": 1, "baron_diff": 1, "herald_diff": 1, "grub_diff": 1,
    "tower_diff": 1, "inhib_diff": 1, "plate_diff": 1,
    "gold_diff": 1000, "xp_diff": 1000, "cs_diff": 10, "level_diff": 1,
}
LABEL = {
    "kill_diff": "champion kill", "dragon_diff": "dragon", "baron_diff": "baron",
    "herald_diff": "rift herald", "grub_diff": "void grub", "tower_diff": "tower",
    "inhib_diff": "inhibitor", "plate_diff": "turret plate",
    "gold_diff": "1000 gold", "xp_diff": "1000 xp", "cs_diff": "10 cs", "level_diff": "1 level",
}


def ame(feat, unit):
    Xp = X.copy()
    Xp[feat] = Xp[feat] + unit
    return (clf.predict_proba(Xp)[:, 1] - base).mean() * 100


def at_even(feat, minute, unit):
    row = {c: 0.0 for c in F}; row["minute"] = minute
    a = clf.predict_proba(pd.DataFrame([row])[F])[0, 1]
    row[feat] += unit
    b = clf.predict_proba(pd.DataFrame([row])[F])[0, 1]
    return (b - a) * 100


print(f"{'event / resource':<16}{'unit':>6}{'AME ΔP% (all states)':>22}{'at even min15':>15}{'min25':>9}")
rows = []
for f, u in UNITS.items():
    a = ame(f, u); e15 = at_even(f, 15, u); e25 = at_even(f, 25, u)
    rows.append((LABEL[f], u, a, e15, e25))
    print(f"{LABEL[f]:<16}{u:>6}{a:>+22.3f}{e15:>+15.3f}{e25:>+9.3f}")

# gold -> win conversion
g_ame = ame("gold_diff", 1000)
g_e20 = at_even("gold_diff", 20, 1000)
print(f"\n== gold -> win conversion ==")
print(f"  +1000 gold: {g_ame:+.3f}% (avg over states) | {g_e20:+.3f}% (even, min20)")
print(f"  => at an even mid-game, ~{1000/abs(g_e20):.0f} gold per +1% win  (~{50/abs(g_e20)*1000/1000:.1f}k gold to swing 50%->100% linearly)")

# express each structural objective in GOLD-EQUIVALENT (value beyond its own gold), via even-min20 ratio
print(f"\n== structural value in GOLD-EQUIVALENT (beyond direct gold; even min20) ==")
for f in ["kill_diff", "tower_diff", "inhib_diff", "plate_diff", "dragon_diff", "herald_diff", "grub_diff", "baron_diff"]:
    v = at_even(f, 20, 1)
    print(f"  {LABEL[f]:<14} {v:+.3f}% win  =  {v/g_e20*1000:>6.0f} gold-equiv")
