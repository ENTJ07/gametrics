"""KR high-elo soloq raw collector. Resumable. seed -> discover -> fetch.

Usage:
  python collect.py seed                         # pull challenger+GM+master ladders into summoner pool
  python collect.py discover --max 500           # crawl match-ids for N undiscovered summoners
  python collect.py fetch --target 2000          # fetch match+timeline until N stored (or pending dry)
  python collect.py stats                         # show progress
"""
import argparse, os, sys, time

from riot_client import RiotClient, RateLimiter
from store import Store


def load_env(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


HERE = os.path.dirname(os.path.abspath(__file__))
ENV = load_env(os.path.join(HERE, ".env"))
DB_PATH = os.path.join(HERE, "data", "lol.sqlite")


def make_client(verbose=True):
    return RiotClient(ENV["RIOT_API_KEY"], ENV.get("PLATFORM", "kr"),
                      ENV.get("REGION", "asia"), RateLimiter(), verbose=verbose)


def cmd_seed(args):
    cl, st = make_client(), Store(DB_PATH)
    total = 0
    for tier in ("challenger", "grandmaster", "master"):
        data = cl.league(tier)
        if not data:
            print(f"  {tier}: no data")
            continue
        entries = data.get("entries", [])
        for e in entries:
            puuid = e.get("puuid")
            if puuid:
                st.add_summoner(puuid, tier.upper(), e.get("rank"), e.get("leaguePoints"),
                                e.get("wins"), e.get("losses"), source="ladder")
                total += 1
        st.commit()
        print(f"  {tier}: +{len(entries)} entries")
    print(f"seed done. summoner pool now: {st.stats()['summoners']} (added {total})")


def cmd_discover(args):
    cl, st = make_client(), Store(DB_PATH)
    puuids = st.undiscovered_summoners(args.max)
    print(f"discover: {len(puuids)} summoners x up to {args.count} matches each")
    start_time = int(time.time()) - args.days * 86400 if args.days else None
    new_ids = 0
    for i, puuid in enumerate(puuids, 1):
        ids = cl.match_ids(puuid, queue=420, count=args.count, start_time=start_time)
        before = st.stats()["match_ids_total"]
        for mid in ids:
            st.add_match_id(mid, src=puuid[:12])
        st.mark_discovered(puuid)
        st.commit()
        added = st.stats()["match_ids_total"] - before
        new_ids += added
        if i % 25 == 0 or i == len(puuids):
            s = st.stats()
            print(f"  [{i}/{len(puuids)}] +{new_ids} new ids | frontier pending={s['match_ids_pending']}")
    print(f"discover done. {st.stats()}")


def _fetch_one(cl, st, mid, snowball, patch=None):
    """Returns True if a match was newly stored. `patch` = version prefix filter e.g. '16.12'."""
    if st.have_match(mid):
        st.set_match_status(mid, "done"); st.commit(); return False
    m = cl.match(mid)
    if m is None:
        st.set_match_status(mid, "skip"); st.commit(); return False
    info = m.get("info", {})
    # filter BEFORE the timeline request to save rate budget on off-target games
    if info.get("queueId") != 420 or (patch and not str(info.get("gameVersion", "")).startswith(patch)):
        st.set_match_status(mid, "skip"); st.commit(); return False
    tl = cl.timeline(mid)
    st.save_match(mid, m)
    if tl is not None:
        st.save_timeline(mid, tl)
    st.set_match_status(mid, "done")
    if snowball:
        for p in info.get("participants", []):
            if p.get("puuid"):
                st.add_summoner(p["puuid"], None, None, None, None, None, source="snowball")
    st.commit()
    return True


def cmd_fetch(args):
    cl, st = make_client(), Store(DB_PATH)
    done, t0 = 0, time.time()
    while True:
        batch = st.pending_match_ids(200)
        if not batch:
            print("no pending match ids left."); break
        for mid in batch:
            if _fetch_one(cl, st, mid, args.snowball, getattr(args, "patch", None)):
                done += 1
                if done % 20 == 0:
                    s = st.stats()
                    print(f"  fetched {done} | stored={s['matches_stored']} pending={s['match_ids_pending']} "
                          f"| ~{done/(time.time()-t0)*3600:.0f}/h")
            if args.target and st.stats()["matches_stored"] >= args.target:
                print(f"reached target {args.target}."); return
    print(f"fetch loop ended. {st.stats()}")


def _resolve_start_time(args):
    if getattr(args, "start_date", None):
        import datetime
        return int(datetime.datetime.strptime(args.start_date, "%Y-%m-%d").timestamp())
    if args.days:
        return int(time.time()) - args.days * 86400
    return None


def cmd_run(args):
    """Self-sustaining: auto-replenish frontier via discover, then fetch, until target/exhaustion."""
    cl, st = make_client(), Store(DB_PATH)
    t0, done = time.time(), 0
    start_time = _resolve_start_time(args)
    print(f"run: patch={args.patch or 'ANY'} start_time={start_time} target={args.target}", flush=True)
    while True:
        s = st.stats()
        if args.target and s["matches_stored"] >= args.target:
            print(f"target {args.target} reached. {s}"); break
        if s["match_ids_pending"] < args.low:
            puuids = st.undiscovered_summoners(args.discover_batch)
            if not puuids and s["match_ids_pending"] == 0:
                print("summoner pool & frontier exhausted."); break
            for puuid in puuids:
                for mid in cl.match_ids(puuid, queue=420, count=args.count, start_time=start_time):
                    st.add_match_id(mid, puuid[:12])
                st.mark_discovered(puuid)
            st.commit()
            print(f"  + replenished: pending={st.stats()['match_ids_pending']} "
                  f"(crawled {st.stats()['summoners_crawled']} summoners)")
        for mid in st.pending_match_ids(100):
            if _fetch_one(cl, st, mid, args.snowball, args.patch):
                done += 1
                if done % 25 == 0:
                    s = st.stats()
                    print(f"  fetched {done} this run | stored={s['matches_stored']:,} "
                          f"pending={s['match_ids_pending']} | ~{done/(time.time()-t0)*3600:.0f}/h", flush=True)
            if args.target and st.stats()["matches_stored"] >= args.target:
                print(f"target {args.target} reached."); return


def cmd_prune(args):
    """Delete stored matches/timelines NOT matching the target patch prefix (patch purity)."""
    st = Store(DB_PATH)
    like = args.patch + "%"
    before = st.stats()["matches_stored"]
    drop_ids = [r[0] for r in st.db.execute(
        "SELECT match_id FROM matches WHERE game_version NOT LIKE ?", (like,))]
    st.db.execute("DELETE FROM timelines WHERE match_id IN (SELECT match_id FROM matches WHERE game_version NOT LIKE ?)", (like,))
    st.db.execute("DELETE FROM matches WHERE game_version NOT LIKE ?", (like,))
    for mid in drop_ids:
        st.set_match_status(mid, "skip")
    st.commit()
    print(f"pruned {len(drop_ids)} off-patch matches (kept {args.patch}). stored {before} -> {st.stats()['matches_stored']}")


def cmd_stats(args):
    st = Store(DB_PATH)
    s = st.stats()
    w = max(len(k) for k in s)
    for k, v in s.items():
        print(f"  {k:<{w}} : {v:,}")
    pb = st.patch_breakdown()
    if pb:
        print("  patches:", ", ".join(f"{p}={n}" for p, n in pb[:8]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    d = sub.add_parser("discover"); d.add_argument("--max", type=int, default=500); d.add_argument("--count", type=int, default=40); d.add_argument("--days", type=int, default=0)
    f = sub.add_parser("fetch"); f.add_argument("--target", type=int, default=0); f.add_argument("--snowball", action="store_true"); f.add_argument("--patch", default=None)
    r = sub.add_parser("run")
    r.add_argument("--target", type=int, default=0)
    r.add_argument("--patch", default=None, help="version prefix filter, e.g. 16.12 (store only these)")
    r.add_argument("--start-date", default=None, help="collect matches on/after this date, YYYY-MM-DD")
    r.add_argument("--low", type=int, default=300, help="replenish frontier when pending drops below this")
    r.add_argument("--discover-batch", type=int, default=200, help="summoners to crawl per replenish")
    r.add_argument("--count", type=int, default=40)
    r.add_argument("--days", type=int, default=0)
    r.add_argument("--snowball", action="store_true")
    p = sub.add_parser("prune"); p.add_argument("--patch", required=True)
    sub.add_parser("stats")
    args = ap.parse_args()
    {"seed": cmd_seed, "discover": cmd_discover, "fetch": cmd_fetch, "run": cmd_run,
     "prune": cmd_prune, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
