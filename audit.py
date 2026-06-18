"""Quality audit of the raw lake before building features. Cheap index checks via SQL,
participant/timeline checks via decompression."""
import sqlite3, statistics as stats
from store import Store

st = Store("data/lol.sqlite")
db = st.db

def pct(x, n): return f"{x:,} ({100*x/n:.1f}%)" if n else "0"

n = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
print(f"matches: {n:,}\n")

print("== index-level (no decompress) ==")
print(" patch:", db.execute("SELECT game_version,COUNT(*) FROM matches GROUP BY substr(game_version,1,5)").fetchall()[:3])
print(" queue:", db.execute("SELECT queue_id,COUNT(*) FROM matches GROUP BY queue_id").fetchall())
durs = [r[0] for r in db.execute("SELECT game_duration FROM matches WHERE game_duration IS NOT NULL")]
durs.sort()
q = lambda p: durs[int(p*len(durs))]
print(f" duration s: min={durs[0]} p5={q(.05)} median={q(.5)} p95={q(.95)} max={durs[-1]}")
print(f" remakes (<300s): {pct(sum(1 for d in durs if d<300), n)}")
print(f" short (<900s): {pct(sum(1 for d in durs if d<900), n)}")

print("\n== participant/timeline-level (decompress full scan) ==")
bad_pcount = empty_pos = early_surr = 0
win_gold = []; lose_gold = []
frame_counts = []; tl_missing = 0
pos_fill = 0; pos_total = 0
ranks_seen = {}
cur = db.execute("SELECT match_id FROM matches")
ids = [r[0] for r in cur]
for i, mid in enumerate(ids):
    m = st.load_match(mid)
    info = m["info"]
    ps = info["participants"]
    if len(ps) != 10:
        bad_pcount += 1
    for p in ps:
        pos_total += 1
        tp = p.get("teamPosition", "")
        if tp:
            pos_fill += 1
        else:
            empty_pos += 1
        (win_gold if p["win"] else lose_gold).append(p["goldEarned"])
        if p.get("gameEndedInEarlySurrender"):
            early_surr += 1
    tl = st.load_timeline(mid)
    if tl is None:
        tl_missing += 1
    else:
        frame_counts.append(len(tl["info"]["frames"]))
    if (i + 1) % 2000 == 0:
        print(f"   ...scanned {i+1:,}/{len(ids):,}")

print(f"\n matches with !=10 participants: {bad_pcount}")
print(f" teamPosition fill rate: {pct(pos_fill, pos_total)}  (empty: {empty_pos:,})")
print(f" early-surrender participant rows: {early_surr:,} (~{early_surr//10:,} games)")
print(f" timelines missing: {tl_missing}")
fc = sorted(frame_counts)
print(f" timeline frames: min={fc[0]} median={fc[len(fc)//2]} max={fc[-1]}")
print(f"\n SANITY win vs lose avg goldEarned: {stats.mean(win_gold):,.0f} vs {stats.mean(lose_gold):,.0f}"
      f"  (delta {stats.mean(win_gold)-stats.mean(lose_gold):,.0f}; winners should be higher)")
