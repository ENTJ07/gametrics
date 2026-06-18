"""WAA -> WAR. Replacement level = role-specific low percentile of per-game contribution rate.
WAR = (player rate - replacement rate) * games  => playing-time-rewarding counting stat."""
import numpy as np
import pandas as pd

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
P = 10          # replacement = 10th percentile rate within role (freely-available level)
MIN_STABLE = 20  # games needed to help DEFINE replacement (stable rate)

agg = pd.read_parquet("data/parsed/player_waa.parquet")
stable = agg[agg.games >= MIN_STABLE]
repl = {r: np.percentile(stable[stable.role == r].waa_per, P) for r in ROLES}
print("replacement rate (wins/game below which = freely available), by role:")
for r in ROLES:
    print(f"  {r:<8} {repl[r]:+.4f}/g   (avg player = 0; replacement is this far below)")

agg["repl_rate"] = agg.role.map(repl)
agg["war"] = (agg.waa_per - agg.repl_rate) * agg.games
agg.to_parquet("data/parsed/player_war.parquet", index=False)

elig = agg[agg.games >= 15].copy()
print(f"\n% eligible players with WAR > 0: {(elig.war > 0).mean():.1%}  (WAR should make almost all regulars positive)")
print(f"mean WAR {elig.war.mean():.2f} | WAR vs winrate corr {elig.war.corr(elig.winrate):.3f}")

# playing-time property: an exactly-average player (rate 0) accrues WAR purely from volume
print("\nWAR of a role-average player (rate=0) by games played:")
for g in [25, 50, 100]:
    war_avg = -np.mean(list(repl.values())) * g
    print(f"  {g:>3} games -> {war_avg:.2f} WAR  (volume alone, no skill above average)")

print(f"\n=== TOP 15 by WAR (>=15 games, {len(elig):,} players) ===")
print(f"{'role':<8}{'games':>6}{'WAR':>7}{'rate':>8}{'winrate':>9}")
for _, r in elig.sort_values("war", ascending=False).head(15).iterrows():
    print(f"{r.role:<8}{r.games:>6.0f}{r.war:>7.2f}{r.waa_per:>+8.3f}{r.winrate:>8.0%}")

print("\nbest WAR per role:")
for role in ROLES:
    rr = elig[elig.role == role].sort_values("war", ascending=False).head(1).iloc[0]
    print(f"  {role:<8} WAR {rr.war:>6.2f}  ({rr.games:.0f} games, rate {rr.waa_per:+.3f}, {rr.winrate:.0%} wr, overall #{(elig.war>rr.war).sum()+1})")

# contrast: does WAR reward volume vs WAA? show a high-volume average player vs low-volume star
print("\nWAA(rate) vs WAR(counting) - Spearman with games:")
print(f"  WAA/g rank-corr w/ games: {elig.waa_per.corr(elig.games, method='spearman'):+.3f}  (skill rate ~ independent of volume)")
print(f"  WAR   rank-corr w/ games: {elig.war.corr(elig.games, method='spearman'):+.3f}  (counting stat rewards volume)")
