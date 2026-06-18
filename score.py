"""On-demand scoring with per-role and per-champion breakdowns.
extract_player_game -> per-game dict; score_games -> overall + roles[] + champions[].
Same aggregator powers both precomputed export and live lookups."""
import os
from collections import Counter
import numpy as np
import joblib

ART = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "artifacts.joblib"))
CHK = ["econ", "combat", "obj", "vision", "cc", "tank"]
MON = {"DRAGON": "dragon_credit", "BARON_NASHOR": "baron_credit", "RIFTHERALD": "herald_credit", "HORDE": "grub_credit"}
BLD = {"TOWER_BUILDING": "tower_credit", "INHIBITOR_BUILDING": "inhib_credit"}


def extract_player_game(match, timeline, puuid):
    info = match["info"]
    if info.get("queueId") != 420 or info.get("gameDuration", 0) < 300:
        return None
    p = next((x for x in info["participants"] if x["puuid"] == puuid), None)
    if p is None or (p.get("teamPosition") or "") not in ART["baselines"]:
        return None
    if any(x.get("gameEndedInEarlySurrender") for x in info["participants"]):
        return None
    pid = p["participantId"]
    q = {"gold_earned": p.get("goldEarned", 0),
         "vision_actions": p.get("wardsPlaced", 0) + p.get("wardsKilled", 0),
         "kill_credit": 0.0, "dragon_credit": 0.0, "baron_credit": 0.0, "herald_credit": 0.0,
         "grub_credit": 0.0, "tower_credit": 0.0, "inhib_credit": 0.0, "plate_credit": 0.0}
    frames = timeline["info"]["frames"]
    for fr in frames:
        for ev in fr.get("events", []):
            t = ev["type"]
            killer = ev.get("killerId", 0) or 0
            assists = ev.get("assistingParticipantIds", []) or []
            if t == "CHAMPION_KILL":
                if killer == pid:
                    q["kill_credit"] += 0.6 if assists else 1.0
                if pid in assists:
                    q["kill_credit"] += 0.4 / len(assists)
            else:
                key = (MON.get(ev.get("monsterType")) if t == "ELITE_MONSTER_KILL"
                       else BLD.get(ev.get("buildingType")) if t == "BUILDING_KILL"
                       else "plate_credit" if t == "TURRET_PLATE_DESTROYED" else None)
                if key:
                    inv = ([killer] if killer >= 1 else []) + assists
                    if inv and pid in inv:
                        q[key] += 1.0 / len(inv)
    last = frames[-1]["participantFrames"].get(str(pid), {})
    q["player_cc"] = last.get("timeEnemySpentControlled", 0)
    q["player_dtaken"] = last.get("damageStats", {}).get("totalDamageTaken", 0)
    craft = {"gold_earned": p.get("goldEarned", 0),
             "cs": p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
             "dmg_champ": p.get("totalDamageDealtToChampions", 0), "vision_score": p.get("visionScore", 0),
             "control_wards": p.get("detectorWardsPlaced", 0), "wards_placed": p.get("wardsPlaced", 0),
             "wards_killed": p.get("wardsKilled", 0), "kill_credit": q["kill_credit"],
             "deaths": p.get("deaths", 0), "player_cc": q["player_cc"], "player_dtaken": q["player_dtaken"]}
    return {"role": p["teamPosition"], "win": bool(p.get("win")), "minutes": max(info["gameDuration"] / 60.0, 1),
            "q": q, "craft": craft, "champion": p.get("championName")}


def _channels(g):
    b, q, r = ART["baselines"][g["role"]], g["q"], g["role"]
    return {"econ": (q["gold_earned"] - b["gold_earned"]) * ART["role_gold_w"][r],
            "combat": (q["kill_credit"] - b["kill_credit"]) * ART["struct_w"]["kill_credit"],
            "obj": sum((q[k] - b[k]) * ART["struct_w"][k] for k in ART["struct_map"] if k != "kill_credit"),
            "vision": (q["vision_actions"] - b["vision_actions"]) * ART["vision_w"],
            "cc": (q["player_cc"] - b["player_cc"]) * ART["cc_w"],
            "tank": (q["player_dtaken"] - b["player_dtaken"]) * ART["tank_w"]}


def _agg(gs):
    chan = {k: 0.0 for k in CHK}
    waa = 0.0
    for g in gs:
        c = _channels(g)
        for k in CHK:
            chan[k] += c[k]
        waa += sum(c.values()) / 100.0
    ab = {k: abs(v) for k, v in chan.items()}
    tot = sum(ab.values()) or 1
    return waa, {k: round(ab[k] / tot * 100) for k in CHK}


def _skill(gs, role):
    sm = ART["skill_models"][role]
    mean, scale, coef = np.array(sm["mean"]), np.array(sm["scale"]), np.array(sm["coef"])
    idxs = [float(((np.array([g["craft"][f] / g["minutes"] for f in sm["feats"]]) - mean) / scale) @ coef)
            for g in gs if g["role"] == role]
    if not idxs:
        return None
    return int(round(np.searchsorted(sm["ref_idx"], np.mean(idxs)) / len(sm["ref_idx"]) * 100))


def _grp_score(gs, role):
    waa, ch = _agg(gs)
    n = len(gs)
    wp = waa / n
    war = (wp - ART["repl_rate"][role]) * n
    wr = round(sum(g["win"] for g in gs) / n * 100)
    return {"role": ART["role_label"][role], "games": n, "winrate": wr, "war": round(war, 1),
            "rate": round(wp, 3), "skill": _skill(gs, role), "ch": ch}


def score_games(gs):
    if not gs:
        return None
    by_role = {}
    for g in gs:
        by_role.setdefault(g["role"], []).append(g)
    roles = sorted([_grp_score(gr, r) for r, gr in by_role.items()], key=lambda x: -x["games"])
    main = roles[0]

    by_ch = {}
    for g in gs:
        by_ch.setdefault(g["champion"], []).append(g)
    champions = []
    for cn, gc in by_ch.items():
        role = Counter(g["role"] for g in gc).most_common(1)[0][0]
        s = _grp_score(gc, role)
        champions.append({"champion": cn or "?", "role": s["role"], "games": s["games"],
                          "winrate": s["winrate"], "war": s["war"], "rate": s["rate"]})
    champions.sort(key=lambda x: -x["games"])

    n = len(gs)
    total_war = round(sum(r["war"] for r in roles), 1)
    _, overall_ch = _agg(gs)
    refw = ART["ref_warall"]
    return {"role": main["role"], "war": total_war,
            "war_rank": int(len(refw) - np.searchsorted(refw, total_war) + 1),
            "skill": main["skill"], "games": n,
            "winrate": round(sum(g["win"] for g in gs) / n * 100),
            "rate": round(sum(r["rate"] * r["games"] for r in roles) / n, 3),
            "champs": [c["champion"] for c in champions[:3]], "ch": overall_ch,
            "prov": n < 15, "roles": roles, "champions": champions}


def score_player(puuid, pairs):
    gs = [e for e in (extract_player_game(m, t, puuid) for m, t in pairs) if e]
    return score_games(gs)
