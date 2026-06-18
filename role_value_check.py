"""Does a gold lead predict winning differently depending on WHICH ROLE holds it?
Standardized logistic coefficients on per-role gold leads -> comparable 'win value per role'."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

part = pd.read_parquet("data/parsed/participants.parquet", columns=["match_id", "participant_id", "team_position", "team_id"])
meta = pd.read_parquet("data/parsed/match_meta.parquet")
good_ids = set(meta[(~meta.is_remake) & (~meta.early_surrender)].match_id)
win100 = meta.set_index("match_id").winning_team.eq(100).astype(int)
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

for MIN in [15, 20, 25]:
    pf = pd.read_parquet("data/parsed/pframes.parquet", columns=["match_id", "minute", "participant_id", "total_gold", "xp"])
    pf = pf[(pf.minute == MIN) & (pf.match_id.isin(good_ids))]
    pf = pf.merge(part, on=["match_id", "participant_id"])
    g = pf.groupby(["match_id", "team_position", "team_id"]).total_gold.sum().reset_index()
    piv = g.pivot_table(index="match_id", columns=["team_position", "team_id"], values="total_gold")
    feat = pd.DataFrame(index=piv.index)
    for r in ROLES:
        if (r, 100) in piv.columns and (r, 200) in piv.columns:
            feat[r] = piv[(r, 100)] - piv[(r, 200)]
    feat = feat.dropna()
    lab = win100.loc[feat.index].values
    Xs = StandardScaler().fit_transform(feat[ROLES].values)
    lr = LogisticRegression(max_iter=1000).fit(Xs, lab)
    auc = roc_auc_score(lab, lr.predict_proba(Xs)[:, 1])
    raw_sd = feat[ROLES].std()
    print(f"\n== minute {MIN}: n={len(feat):,}  AUC={auc:.3f}  (std logistic coef = win value per SD of that role's gold lead) ==")
    order = sorted(zip(ROLES, lr.coef_[0]), key=lambda t: -t[1])
    for r, c in order:
        print(f"   {r:<8} stdcoef {c:+.3f}   (1 SD = {raw_sd[r]:,.0f} gold)")
