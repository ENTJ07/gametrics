"""One-shot probe: verify key + lock the actual response schema before building the collector."""
import json, os, sys, time
import requests

def load_env(path=".env"):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

ENV = load_env(os.path.join(os.path.dirname(__file__), ".env"))
KEY = ENV["RIOT_API_KEY"]
PLATFORM = ENV.get("PLATFORM", "kr")
REGION = ENV.get("REGION", "asia")
H = {"X-Riot-Token": KEY}

def get(url, **params):
    r = requests.get(url, headers=H, params=params, timeout=15)
    print(f"  -> {r.status_code} {url.split('riotgames.com')[-1][:80]}")
    if r.status_code != 200:
        print("     headers:", {k: v for k, v in r.headers.items() if "Rate" in k or "Retry" in k})
        print("     body:", r.text[:300])
        return None
    return r.json()

print("== 1. Challenger ladder ==")
chal = get(f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5")
if not chal:
    sys.exit("key/endpoint failed")
entries = chal.get("entries", [])
print(f"  entries: {len(entries)}")
print(f"  entry[0] keys: {sorted(entries[0].keys())}")
print(f"  entry[0] sample: {json.dumps({k: entries[0][k] for k in entries[0]}, ensure_ascii=False)[:300]}")

e0 = entries[0]
puuid = e0.get("puuid")
if not puuid and e0.get("summonerId"):
    print("\n== 1b. No puuid in league entry -> SUMMONER-V4 lookup ==")
    s = get(f"https://{PLATFORM}.api.riotgames.com/lol/summoner/v4/summoners/{e0['summonerId']}")
    if s:
        print(f"  summoner keys: {sorted(s.keys())}")
        puuid = s.get("puuid")
print(f"\n  resolved puuid: {puuid[:16] if puuid else None}...")

print("\n== 2. Match IDs (queue 420 = ranked solo) ==")
ids = get(f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids", queue=420, start=0, count=5)
print(f"  match ids: {ids}")

if ids:
    mid = ids[0]
    print(f"\n== 3. Match detail {mid} ==")
    m = get(f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{mid}")
    if m:
        info = m["info"]
        print(f"  gameVersion: {info.get('gameVersion')}  queueId: {info.get('queueId')}  duration: {info.get('gameDuration')}s")
        print(f"  info keys: {sorted(info.keys())}")
        p0 = info["participants"][0]
        print(f"  participant count: {len(info['participants'])}")
        print(f"  participant[0] field count: {len(p0)}")
        print(f"  participant[0] keys: {sorted(p0.keys())}")

    print(f"\n== 4. Timeline {mid} ==")
    t = get(f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{mid}/timeline")
    if t:
        frames = t["info"]["frames"]
        print(f"  frames: {len(frames)}  (interval={t['info'].get('frameInterval')}ms)")
        f1 = frames[1]
        print(f"  frame keys: {sorted(f1.keys())}")
        pf = f1["participantFrames"]["1"]
        print(f"  participantFrame[1] keys: {sorted(pf.keys())}")
        ev_types = {}
        for fr in frames:
            for ev in fr.get("events", []):
                ev_types[ev["type"]] = ev_types.get(ev["type"], 0) + 1
        print(f"  event types in this game: {json.dumps(ev_types, ensure_ascii=False)}")
