"""SQLite raw data lake. Stores lossless gzipped match/timeline JSON + crawl frontier."""
import sqlite3, gzip, json, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS summoners (
  puuid TEXT PRIMARY KEY,
  tier TEXT, division TEXT, league_points INTEGER,
  wins INTEGER, losses INTEGER,
  source TEXT,
  discovered INTEGER DEFAULT 0,
  added_at REAL
);
CREATE INDEX IF NOT EXISTS idx_sum_disc ON summoners(discovered);

CREATE TABLE IF NOT EXISTS match_ids (
  match_id TEXT PRIMARY KEY,
  status TEXT DEFAULT 'pending',          -- pending / done / skip / failed
  discovered_from TEXT,
  discovered_at REAL
);
CREATE INDEX IF NOT EXISTS idx_match_status ON match_ids(status);

CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY,
  game_version TEXT, queue_id INTEGER,
  game_creation INTEGER, game_duration INTEGER,
  data BLOB, collected_at REAL
);

CREATE TABLE IF NOT EXISTS timelines (
  match_id TEXT PRIMARY KEY,
  data BLOB, collected_at REAL
);
"""


def _pack(obj):
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _unpack(blob):
    return json.loads(gzip.decompress(blob).decode("utf-8"))


class Store:
    def __init__(self, path="data/lol.sqlite"):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- summoners ---
    def add_summoner(self, puuid, tier, division, lp, wins, losses, source):
        self.db.execute(
            "INSERT OR IGNORE INTO summoners(puuid,tier,division,league_points,wins,losses,source,added_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (puuid, tier, division, lp, wins, losses, source, time.time()),
        )

    def undiscovered_summoners(self, limit):
        return [r[0] for r in self.db.execute(
            "SELECT puuid FROM summoners WHERE discovered=0 LIMIT ?", (limit,))]

    def mark_discovered(self, puuid):
        self.db.execute("UPDATE summoners SET discovered=1 WHERE puuid=?", (puuid,))

    # --- match frontier ---
    def add_match_id(self, match_id, src):
        self.db.execute(
            "INSERT OR IGNORE INTO match_ids(match_id,discovered_from,discovered_at) VALUES(?,?,?)",
            (match_id, src, time.time()))

    def pending_match_ids(self, limit):
        return [r[0] for r in self.db.execute(
            "SELECT match_id FROM match_ids WHERE status='pending' LIMIT ?", (limit,))]

    def set_match_status(self, match_id, status):
        self.db.execute("UPDATE match_ids SET status=? WHERE match_id=?", (status, match_id))

    def have_match(self, match_id):
        return self.db.execute("SELECT 1 FROM matches WHERE match_id=?", (match_id,)).fetchone() is not None

    # --- raw payloads ---
    def save_match(self, match_id, match_json):
        info = match_json.get("info", {})
        self.db.execute(
            "INSERT OR REPLACE INTO matches(match_id,game_version,queue_id,game_creation,game_duration,data,collected_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (match_id, info.get("gameVersion"), info.get("queueId"),
             info.get("gameCreation"), info.get("gameDuration"),
             _pack(match_json), time.time()))

    def save_timeline(self, match_id, tl_json):
        self.db.execute(
            "INSERT OR REPLACE INTO timelines(match_id,data,collected_at) VALUES(?,?,?)",
            (match_id, _pack(tl_json), time.time()))

    def load_match(self, match_id):
        row = self.db.execute("SELECT data FROM matches WHERE match_id=?", (match_id,)).fetchone()
        return _unpack(row[0]) if row else None

    def load_timeline(self, match_id):
        row = self.db.execute("SELECT data FROM timelines WHERE match_id=?", (match_id,)).fetchone()
        return _unpack(row[0]) if row else None

    def commit(self):
        self.db.commit()

    # --- stats ---
    def stats(self):
        c = self.db.execute
        def one(q): return c(q).fetchone()[0]
        return {
            "summoners": one("SELECT COUNT(*) FROM summoners"),
            "summoners_crawled": one("SELECT COUNT(*) FROM summoners WHERE discovered=1"),
            "match_ids_total": one("SELECT COUNT(*) FROM match_ids"),
            "match_ids_pending": one("SELECT COUNT(*) FROM match_ids WHERE status='pending'"),
            "match_ids_done": one("SELECT COUNT(*) FROM match_ids WHERE status='done'"),
            "matches_stored": one("SELECT COUNT(*) FROM matches"),
            "timelines_stored": one("SELECT COUNT(*) FROM timelines"),
        }

    def patch_breakdown(self):
        return self.db.execute(
            "SELECT substr(game_version,1,5), COUNT(*) FROM matches GROUP BY 1 ORDER BY 2 DESC").fetchall()
