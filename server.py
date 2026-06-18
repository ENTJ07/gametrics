"""On-demand evaluation API + static site. GET /api/evaluate?name=&tag=
Precomputed players return instantly; unknown players are scored live via Riot API."""
import os, time, json, datetime
from urllib.parse import quote
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from riot_client import RiotClient, RateLimiter
from score import score_player

HERE = os.path.dirname(os.path.abspath(__file__))
def load_env(p):
    e = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); e[k.strip()] = v.strip()
    return e
_envfile = load_env(os.path.join(HERE, ".env")) if os.path.exists(os.path.join(HERE, ".env")) else {}
def _cfg(k, d=None):
    return os.environ.get(k) or _envfile.get(k, d)   # env var (Render) overrides .env (local)
KEY = _cfg("RIOT_API_KEY")
if not KEY:
    raise RuntimeError("RIOT_API_KEY not set (set env var on host, or .env locally)")
REGION, PLATFORM = _cfg("REGION", "asia"), _cfg("PLATFORM", "kr")
cl = RiotClient(KEY, PLATFORM, REGION, RateLimiter(), verbose=False)

PATCH = "16.12"
START = int(datetime.datetime(2026, 6, 10).timestamp())
MAXG = 20

PRE = {}
try:
    for p in json.load(open(os.path.join(HERE, "web", "data.json"), encoding="utf-8"))["players"]:
        PRE[(p["name"].lower(), p["tag"].lower())] = p
except FileNotFoundError:
    pass
CACHE, TTL = {}, 3600

app = FastAPI(title="Gametrics")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/evaluate")
def evaluate(name: str, tag: str):
    key = (name.lower().strip(), tag.lower().strip())
    if key in PRE:
        return {**PRE[key], "source": "precomputed"}
    try:
        acc = cl._get(f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{quote(name)}/{quote(tag)}")
        if not acc:
            raise HTTPException(404, "Riot ID를 찾을 수 없습니다 (닉변했을 수 있음)")
        puuid = acc["puuid"]
        if puuid in CACHE and time.time() - CACHE[puuid][0] < TTL:
            return {**CACHE[puuid][1], "source": "cache"}
        ids = cl.match_ids(puuid, queue=420, count=MAXG, start_time=START)
        pairs = []
        for mid in ids:
            m = cl.match(mid)
            if not m or not str(m["info"].get("gameVersion", "")).startswith(PATCH):
                continue
            t = cl.timeline(mid)
            if t:
                pairs.append((m, t))
    except RuntimeError:
        raise HTTPException(503, "Riot API 키 오류 (dev 키 만료 가능). 키를 갱신해주세요.")
    if not pairs:
        raise HTTPException(404, f"최근 패치({PATCH}) 랭크 게임이 없습니다")
    r = score_player(puuid, pairs)
    if not r:
        raise HTTPException(404, "채점 가능한 게임이 없습니다")
    r.update(name=acc.get("gameName", name), tag=acc.get("tagLine", tag), source="live")
    CACHE[puuid] = (time.time(), r)
    return r

app.mount("/", StaticFiles(directory=os.path.join(HERE, "web"), html=True), name="static")
