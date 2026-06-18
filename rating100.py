"""0-100 'WAR Score' with empirical-Bayes shrinkage (regress noisy small samples to role mean).
shrunk_rate = role_mean + (raw_rate - role_mean) * games/(games+K), K = noise/skill variance ratio."""
import numpy as np
import pandas as pd

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
pg = pd.read_parquet("data/parsed/player_game_waa.parquet")
war = pd.read_parquet("data/parsed/player_war.parquet")

# per-player rate + games
rate = pg.groupby("puuid").agg(raw=("waa_game", "mean"), games=("waa_game", "size"),
                               role=("team_position", lambda s: s.mode().iloc[0])).reset_index()

# estimate variance components PER ROLE
sigma2, var_true, K, role_mean = {}, {}, {}, {}
for r in ROLES:
    ids = rate[rate.role == r].puuid
    sub = pg[pg.puuid.isin(ids)]
    # within-player game variance (noise), from players with >=10 games
    wv = sub.groupby("puuid").waa_game.var()
    cnt = sub.groupby("puuid").waa_game.size()
    sigma2[r] = wv[cnt >= 10].mean()
    rr = rate[rate.role == r]
    role_mean[r] = (rr.raw * rr.games).sum() / rr.games.sum()
    obs_var = rr.raw.var()
    var_true[r] = max(obs_var - sigma2[r] * (1 / rr.games).mean(), 1e-5)
    K[r] = sigma2[r] / var_true[r]

print("regression constant K (games to half-stabilize the rating), by role:")
for r in ROLES:
    print(f"  {r:<8} K={K[r]:>5.0f} games   (sigma2_game={sigma2[r]:.4f}, var_skill={var_true[r]:.5f})")

rate["K"] = rate.role.map(K)
rate["rm"] = rate.role.map(role_mean)
rate["shrunk"] = rate.rm + (rate.raw - rate.rm) * rate.games / (rate.games + rate.K)

m = rate.merge(war[["puuid", "war", "winrate"]], on="puuid", how="left")
m["score_role"] = m.groupby("role").shrunk.rank(pct=True).mul(100)
m["score_all"] = m.shrunk.rank(pct=True).mul(100)
m.to_parquet("data/parsed/player_rating.parquet", index=False)

elig = m[m.games >= 15].copy()
print(f"\nscore vs winrate Spearman: {elig.score_role.corr(elig.winrate, method='spearman'):.3f} "
      f"(raw was 0.31; shrinkage should hold or improve validity)")
print(f"score_all corr with games: {elig.score_all.corr(elig.games, method='spearman'):+.3f} "
      f"(now POSITIVE = proven volume rewarded, not small-sample noise)")

print(f"\n=== TOP 15 by WAR Score (shrunk skill, single ladder) ===")
print(f"{'role':<8}{'score':>6}{'(role)':>8}{'games':>6}{'raw':>8}{'shrunk':>8}{'winrate':>9}")
for _, r in elig.sort_values("score_all", ascending=False).head(15).iterrows():
    print(f"{r.role:<8}{r.score_all:>6.0f}{r.score_role:>8.0f}{r.games:>6.0f}{r.raw:>+8.3f}{r.shrunk:>+8.3f}{r.winrate:>8.0%}")

print("\nbest per role (score/100 within role):")
for r in ROLES:
    rr = elig[elig.role == r].sort_values("score_role", ascending=False).iloc[0]
    print(f"  {r:<8} {rr.score_role:>3.0f}/100  ({rr.games:.0f} games, raw {rr.raw:+.3f} -> shrunk {rr.shrunk:+.3f}, {rr.winrate:.0%} wr)")
