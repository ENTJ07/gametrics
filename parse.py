"""Raw lake (SQLite) -> tidy parquet tables. One decompression pass emits all 4 tables.

Usage:
  python parse.py --limit 300     # sample for schema check
  python parse.py                 # full
"""
import argparse, json, os
import pandas as pd
from store import Store

OUT = "data/parsed"
os.makedirs(OUT, exist_ok=True)

# curated challenges to lift into participants (rate/quality metrics)
CHALLENGE_KEYS = [
    "killParticipation", "kda", "teamDamagePercentage", "damageTakenOnTeamPercentage",
    "goldPerMinute", "damagePerMinute", "visionScorePerMinute", "visionScoreAdvantageLaneOpponent",
    "soloKills", "turretPlatesTaken", "dragonTakedowns", "baronTakedowns", "riftHeraldTakedowns",
    "controlWardsPlaced", "stealthWardsPlaced", "wardTakedowns", "killsNearEnemyTurret",
    "laneMinionsFirst10Minutes", "maxLevelLeadLaneOpponent", "effectiveHealAndShielding",
    "saveAllyFromDeath", "skillshotsHit", "enemyChampionImmobilizations",
]


def team_of(pid):
    return 100 if pid <= 5 else 200


def parse(limit=None):
    st = Store("data/lol.sqlite")
    ids = [r[0] for r in st.db.execute("SELECT match_id FROM matches ORDER BY game_creation" + (f" LIMIT {limit}" if limit else ""))]
    meta_rows, part_rows, pframe_rows, event_rows = [], [], [], []

    for i, mid in enumerate(ids):
        m = st.load_match(mid)
        info = m["info"]
        ps = info["participants"]
        dur = info.get("gameDuration", 0)
        win_team = next((t["teamId"] for t in info["teams"] if t.get("win")), None)
        early = any(p.get("gameEndedInEarlySurrender") for p in ps)
        surr = any(p.get("gameEndedInSurrender") for p in ps)
        meta_rows.append(dict(
            match_id=mid, game_version=info.get("gameVersion"), game_creation=info.get("gameCreation"),
            game_duration=dur, winning_team=win_team, is_remake=dur < 300, early_surrender=early, surrender=surr,
        ))

        for p in ps:
            ch = p.get("challenges", {}) or {}
            row = dict(
                match_id=mid, participant_id=p["participantId"], puuid=p.get("puuid"),
                team_id=p["teamId"], team_position=p.get("teamPosition"), champion=p.get("championName"),
                champion_id=p.get("championId"), win=p["win"],
                kills=p["kills"], deaths=p["deaths"], assists=p["assists"],
                gold_earned=p["goldEarned"], gold_spent=p.get("goldSpent"),
                cs=p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                champ_level=p.get("champLevel"), champ_xp=p.get("champExperience"),
                dmg_champ=p.get("totalDamageDealtToChampions"), dmg_taken=p.get("totalDamageTaken"),
                dmg_mitigated=p.get("damageSelfMitigated"), dmg_objectives=p.get("damageDealtToObjectives"),
                dmg_turrets=p.get("damageDealtToTurrets"), dmg_buildings=p.get("damageDealtToBuildings"),
                heal_teammates=p.get("totalHealsOnTeammates"), shield_teammates=p.get("totalDamageShieldedOnTeammates"),
                vision_score=p.get("visionScore"), wards_placed=p.get("wardsPlaced"), wards_killed=p.get("wardsKilled"),
                control_wards=p.get("detectorWardsPlaced"), time_cc_others=p.get("timeCCingOthers"),
                total_cc_dealt=p.get("totalTimeCCDealt"), turret_takedowns=p.get("turretTakedowns"),
                dragon_kills=p.get("dragonKills"), baron_kills=p.get("baronKills"),
                objectives_stolen=p.get("objectivesStolen"), time_dead=p.get("totalTimeSpentDead"),
            )
            for k in CHALLENGE_KEYS:
                row["ch_" + k] = ch.get(k)
            part_rows.append(row)

        # timeline
        tl = st.load_timeline(mid)
        if tl is not None:
            for fr in tl["info"]["frames"]:
                minute = fr["timestamp"] // 60000
                for pid_str, pf in fr["participantFrames"].items():
                    pid = int(pid_str)
                    ds = pf.get("damageStats", {})
                    pos = pf.get("position", {})
                    pframe_rows.append(dict(
                        match_id=mid, minute=minute, participant_id=pid, team_id=team_of(pid),
                        total_gold=pf.get("totalGold"), current_gold=pf.get("currentGold"),
                        xp=pf.get("xp"), level=pf.get("level"),
                        minions=pf.get("minionsKilled"), jungle=pf.get("jungleMinionsKilled"),
                        x=pos.get("x"), y=pos.get("y"), time_enemy_controlled=pf.get("timeEnemySpentControlled"),
                        dmg_champ=ds.get("totalDamageDoneToChampions"), dmg_taken=ds.get("totalDamageTaken"),
                    ))
                for ev in fr.get("events", []):
                    t = ev["type"]
                    if t in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO",
                             "SKILL_LEVEL_UP", "LEVEL_UP", "PAUSE_END"):
                        continue  # high-volume, not needed for WAR currency
                    pos = ev.get("position", {}) or {}
                    event_rows.append(dict(
                        match_id=mid, timestamp=ev.get("timestamp"), minute=(ev.get("timestamp", 0)) // 60000,
                        type=t, killer_id=ev.get("killerId"), victim_id=ev.get("victimId"),
                        assists=",".join(map(str, ev.get("assistingParticipantIds", []))) or None,
                        creator_id=ev.get("creatorId"), team_id=ev.get("teamId") or ev.get("killerTeamId"),
                        x=pos.get("x"), y=pos.get("y"),
                        bounty=ev.get("bounty"), shutdown_bounty=ev.get("shutdownBounty"),
                        monster_type=ev.get("monsterType"), monster_sub_type=ev.get("monsterSubType"),
                        building_type=ev.get("buildingType"), tower_type=ev.get("towerType"),
                        lane_type=ev.get("laneType"), ward_type=ev.get("wardType"),
                        kill_type=ev.get("killType"), multi_kill=ev.get("multiKillLength"),
                        name=ev.get("name"),
                    ))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i+1:,}/{len(ids):,}")

    tag = f"_s{limit}" if limit else ""
    for name, rows in [("match_meta", meta_rows), ("participants", part_rows),
                       ("pframes", pframe_rows), ("events", event_rows)]:
        df = pd.DataFrame(rows)
        path = f"{OUT}/{name}{tag}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {name}: {len(df):,} rows, {df.shape[1]} cols -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    parse(ap.parse_args().limit)
