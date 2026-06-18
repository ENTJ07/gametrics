"""Rate-limited Riot API client (KR platform, ASIA regional routing)."""
import time
from collections import deque
import requests


class RateLimiter:
    """Enforces both app limits locally with headroom: 20/1s and 100/120s."""

    def __init__(self, max_1s=18, max_2min=95):
        self.max_1s = max_1s
        self.max_2min = max_2min
        self.times = deque()

    def acquire(self):
        while True:
            now = time.monotonic()
            while self.times and now - self.times[0] > 120:
                self.times.popleft()
            wait = 0.0
            if len(self.times) >= self.max_2min:
                wait = max(wait, 120 - (now - self.times[0]) + 0.05)
            cnt_1s, oldest_1s = 0, None
            for t in reversed(self.times):
                if now - t <= 1.0:
                    cnt_1s += 1
                    oldest_1s = t
                else:
                    break
            if cnt_1s >= self.max_1s and oldest_1s is not None:
                wait = max(wait, 1.0 - (now - oldest_1s) + 0.02)
            if wait <= 0:
                self.times.append(now)
                return
            time.sleep(wait)


class RiotClient:
    def __init__(self, key, platform="kr", region="asia", limiter=None, verbose=True):
        self.platform = platform
        self.region = region
        self.s = requests.Session()
        self.s.headers["X-Riot-Token"] = key
        self.limiter = limiter or RateLimiter()
        self.verbose = verbose

    def _get(self, url, params=None, tries=6):
        for attempt in range(tries):
            self.limiter.acquire()
            try:
                r = self.s.get(url, params=params, timeout=20)
            except requests.RequestException as e:
                if self.verbose:
                    print(f"    net error: {e}; retry in {2**attempt}s")
                time.sleep(2 ** attempt)
                continue
            sc = r.status_code
            if sc == 200:
                return r.json()
            if sc == 429:
                retry = int(r.headers.get("Retry-After", "5"))
                if self.verbose:
                    print(f"    429 rate-limited -> sleep {retry+1}s")
                time.sleep(retry + 1)
                continue
            if sc in (500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt))
                continue
            if sc == 404:
                return None
            if sc in (401, 403):
                raise RuntimeError(f"Auth error {sc} (key invalid/expired?): {r.text[:200]}")
            if self.verbose:
                print(f"    {sc}: {r.text[:150]}")
            time.sleep(2 ** attempt)
        return None

    # --- endpoints ---
    def league(self, tier):
        """tier in {'challenger','grandmaster','master'}"""
        url = f"https://{self.platform}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/RANKED_SOLO_5x5"
        return self._get(url)

    def match_ids(self, puuid, queue=420, count=50, start=0, start_time=None):
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"queue": queue, "count": count, "start": start}
        if start_time:
            params["startTime"] = start_time
        return self._get(url, params) or []

    def match(self, match_id):
        return self._get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}")

    def timeline(self, match_id):
        return self._get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline")
