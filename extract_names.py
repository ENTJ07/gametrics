"""Extract Riot IDs (already present in match data) + top champions per puuid. No API calls."""
import pandas as pd
from collections import defaultdict, Counter
from store import Store

st = Store("data/lol.sqlite")
ids = [r[0] for r in st.db.execute("SELECT match_id FROM matches")]
latest = {}          # puuid -> (gameCreation, name, tag)
champs = defaultdict(Counter)
for i, mid in enumerate(ids):
    m = st.load_match(mid)
    gc = m["info"].get("gameCreation", 0)
    for p in m["info"]["participants"]:
        pu = p["puuid"]
        nm, tg = p.get("riotIdGameName", ""), p.get("riotIdTagline", "")
        if pu not in latest or gc > latest[pu][0]:
            latest[pu] = (gc, nm, tg)
        champs[pu][p.get("championName", "")] += 1
    if (i + 1) % 2500 == 0:
        print(f"  {i+1:,}/{len(ids):,}")

rows = []
for pu, (gc, nm, tg) in latest.items():
    top = [c for c, _ in champs[pu].most_common(3) if c]
    rows.append((pu, nm, tg, ";".join(top)))
df = pd.DataFrame(rows, columns=["puuid", "name", "tag", "champs"])
df.to_parquet("data/parsed/names.parquet", index=False)
print(f"named {len(df):,} players")
