#!/usr/bin/env python3
"""apex-lean: time-budgeted mechfight/1 bot.

Built on the 4.17-scoring bot, with changes aimed at the things that lose matches:
- Hard attention to the 50 ms budget: cached line-of-sight, adaptive sampling, slow-mode fallback.
- Gun only fires when the torso is really on target (tolerance from target size / distance).
- Mortar only overrides aim when the gun can't do the job (no LOS, very long range, gun empty).
- Laser is only used on mechs, and is released instead of being held while blind.
- Retreats (still facing the enemy, so rear armor is never exposed) when 3+ mechs are close or HP is low,
  and never kites out of the zone.
- Telemetry on stderr every 10 s (hp, alive, mode, act() timing) so practice logs show why we died.
"""
from __future__ import annotations

import array
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def dist2d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3d(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def point_in_ring(x: float, y: float, ring: List[float]) -> bool:
    inside = False
    n = len(ring)
    if n < 6:
        return False
    px1, py1 = ring[n - 2], ring[n - 1]
    for i in range(0, n, 2):
        px2, py2 = ring[i], ring[i + 1]
        if (py1 > y) != (py2 > y):
            x_int = (px2 - px1) * (y - py1) / (py2 - py1 + 1e-12) + px1
            if x < x_int:
                inside = not inside
        px1, py1 = px2, py2
    return inside


class WorldMap:
    """Spatial hash over world.json prisms plus the terrain heightfield."""

    def __init__(self, map_dir: str, files: List[str]) -> None:
        self.bounds = {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0}
        self.prisms: List[Dict[str, Any]] = []
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        self.cs = 32.0
        self.terrain: Optional[array.array] = None
        self.t_min_x, self.t_min_y, self.t_step = -513.0, -334.0, 1.0
        self.t_nx, self.t_ny, self.t_scale = 1070, 724, 0.01

        world_path = os.path.join(map_dir, "v2/world.json")
        terrain_path = os.path.join(map_dir, "v2/terrain.i16")
        for f in files:
            if f.endswith("world.json"):
                world_path = os.path.join(map_dir, f)
            elif f.endswith("terrain.i16"):
                terrain_path = os.path.join(map_dir, f)

        if os.path.isfile(world_path):
            try:
                with open(world_path, "rb") as fp:
                    w = json.load(fp)
                if "bounds" in w:
                    self.bounds = w["bounds"]
                t = w.get("terrain") or {}
                self.t_min_x = float(t.get("minX", self.t_min_x))
                self.t_min_y = float(t.get("minY", self.t_min_y))
                self.t_step = float(t.get("step", self.t_step))
                self.t_nx = int(t.get("nx", self.t_nx))
                self.t_ny = int(t.get("ny", self.t_ny))
                self.t_scale = float(t.get("scale", self.t_scale))
                for p in w.get("prisms") or []:
                    ring = p.get("ring") or []
                    if len(ring) < 6:
                        continue
                    xs, ys = ring[0::2], ring[1::2]
                    bb = (min(xs), min(ys), max(xs), max(ys))
                    idx = len(self.prisms)
                    self.prisms.append({
                        "kind": p.get("kind", "building"),
                        "base": float(p.get("base", -10.0)),
                        "top": float(p.get("top", 20.0)),
                        "ring": ring,
                        "holes": p.get("holes") or [],
                        "bbox": bb,
                    })
                    for gx in range(int(bb[0] // self.cs), int(bb[2] // self.cs) + 1):
                        for gy in range(int(bb[1] // self.cs), int(bb[3] // self.cs) + 1):
                            self.grid.setdefault((gx, gy), []).append(idx)
                del w
            except Exception as exc:
                mf.log(f"[map] world.json load problem: {exc}")

        if os.path.isfile(terrain_path):
            try:
                arr = array.array("h")
                with open(terrain_path, "rb") as fp:
                    arr.frombytes(fp.read())
                if sys.byteorder == "big":
                    arr.byteswap()
                self.terrain = arr
            except Exception as exc:
                mf.log(f"[map] terrain load problem: {exc}")

    def terrain_z(self, x: float, y: float) -> float:
        t = self.terrain
        if t is None:
            return 0.0
        fx = clamp((x - self.t_min_x) / self.t_step, 0.0, self.t_nx - 1.001)
        fy = clamp((y - self.t_min_y) / self.t_step, 0.0, self.t_ny - 1.001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        b = j * self.t_nx + i
        if b + self.t_nx + 1 >= len(t):
            return 0.0
        h0 = t[b] * (1.0 - tx) + t[b + 1] * tx
        h1 = t[b + self.t_nx] * (1.0 - tx) + t[b + self.t_nx + 1] * tx
        return (h0 * (1.0 - ty) + h1 * ty) * self.t_scale

    def is_blocked_point(self, x: float, y: float, z: float, radius: float = 2.6,
                         height: float = 9.0) -> Tuple[bool, float]:
        b = self.bounds
        if x - radius < b["minX"] or x + radius > b["maxX"] or y - radius < b["minY"] or y + radius > b["maxY"]:
            return True, 999.0
        cands = self.grid.get((int(x // self.cs), int(y // self.cs)))
        if not cands:
            return False, z
        pts = ((x, y), (x + radius, y), (x - radius, y), (x, y + radius), (x, y - radius))
        for idx in cands:
            p = self.prisms[idx]
            x0, y0, x1, y1 = p["bbox"]
            if x + radius < x0 or x - radius > x1 or y + radius < y0 or y - radius > y1:
                continue
            if z + height <= p["base"] + 0.2 or z >= p["top"] - 1.1:
                continue
            for px, py in pts:
                if x0 <= px <= x1 and y0 <= py <= y1 and point_in_ring(px, py, p["ring"]):
                    if not any(point_in_ring(px, py, h) for h in p["holes"]):
                        return True, p["top"]
        return False, z

    def _solid_at(self, x: float, y: float, z: float) -> bool:
        cands = self.grid.get((int(x // self.cs), int(y // self.cs)))
        if not cands:
            return False
        for idx in cands:
            p = self.prisms[idx]
            x0, y0, x1, y1 = p["bbox"]
            if x < x0 or x > x1 or y < y0 or y > y1 or z < p["base"] or z > p["top"]:
                continue
            if point_in_ring(x, y, p["ring"]) and not any(point_in_ring(x, y, h) for h in p["holes"]):
                return True
        return False

    def has_los(self, a, b, spacing: float = 5.0) -> bool:
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = max(3, min(40, int(math.hypot(dx, dy) / spacing)))
        for s in range(1, n):
            t = s / n
            if self._solid_at(a[0] + dx * t, a[1] + dy * t, a[2] + dz * t):
                return False
        return True


class Bot:
    name = "apex-lean"

    def __init__(self, hello: mf.Hello) -> None:
        self.slot = int(hello.get("slot", 0))
        self.rng = random.Random(int(hello.get("seed", 42)))
        self.cfg = hello.get("config") or {}
        mi = hello.get("map") or {}
        self.world = WorldMap(mi.get("dir", "/arena"), mi.get("files") or [])
        self.budget = float((self.cfg.get("bot") or {}).get("tick_budget_ms", 50.0)) / 1000.0

        self.upgrade_defs: Dict[str, Dict[str, Any]] = {}
        for u in self.cfg.get("upgrades") or []:
            if "id" in u:
                self.upgrade_defs[u["id"]] = u

        self.last_attacker_bearing: Optional[float] = None
        self.last_attacker_tick = -999
        self.last_sound_bearing: Optional[float] = None
        self.last_sound_tick = -999
        self.roam_angle = self.rng.uniform(-math.pi, math.pi)
        self.stuck_ticks = 0
        self.last_pos = [0.0, 0.0, 0.0]
        self.failed_pickups: Dict[int, int] = {}

        self.slow_until = -1
        self.slow = False
        self.times: List[float] = []
        self.late = 0
        self.last_dmg: Tuple[str, float, bool] = ("", 0.0, False)
        self._los_cache: Dict[Any, bool] = {}
        self._muzzle = 7.0
        self._errors = 0
        self.mode = "init"

    # ------------------------------------------------------------ LOS with per-tick cache
    def _los(self, me_pos, m: Dict[str, Any], h: float = 4.5) -> bool:
        key = m.get("id")
        r = self._los_cache.get(key)
        if r is None:
            p = m["pos"]
            r = self.world.has_los(
                [me_pos[0], me_pos[1], me_pos[2] + self._muzzle],
                [p[0], p[1], p[2] + h],
                9.0 if self.slow else 5.0,
            )
            self._los_cache[key] = r
        return r

    # ------------------------------------------------------------ upgrades
    def _pick_upgrade(self, obs: mf.Observation) -> int:
        offer = obs.get("offer")
        choices = (offer or {}).get("choices") or []
        if not choices:
            return 0
        held = {w["kind"] for w in (obs["self"].get("weapons") or []) if w}
        best_i, best_s = 0, -1e9
        for i, c in enumerate(choices):
            cid = str(c.get("id", ""))
            text = f"{cid} {c.get('desc', '')}".lower()
            udef = self.upgrade_defs.get(cid, {})
            s = 0.0
            req = udef.get("requires")
            if req:
                s += 4.5 if req in held else -6.0
            for eff in udef.get("effects") or []:
                path = str(eff.get("path", "")).lower()
                if any(k in path for k in ("armor", "max_hp", "regen")):
                    s += 5.0
                elif any(k in path for k in ("damage", "cooldown", "charge", "reload", "speed")):
                    s += 4.2
                elif any(k in path for k in ("energy", "turn_rate")):
                    s += 3.6
                elif any(k in path for k in ("vision", "near_range", "fov")):
                    s += 2.5
            if any(k in text for k in ("armor", "damage taken", "hp", "regen")):
                s += 3.0
            if any(k in text for k in ("damage", "fire rate", "cooldown", "charge", "chain")):
                s += 2.5
            if any(k in text for k in ("speed", "energy", "turn", "boost")):
                s += 1.5
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    # ------------------------------------------------------------ looting
    def _evaluate_loot(self, obs: mf.Observation) -> Tuple[Optional[int], Optional[int]]:
        me = obs["self"]
        x, y, z = me["pos"]
        tick = int(obs["tick"])
        weapons = me.get("weapons") or [None, None]
        for w in weapons:
            if w and w.get("kind") == "laser" and float(w.get("charge", 0.0)) > 0.0:
                return None, None
        reach = float((self.cfg.get("loot") or {}).get("pickup_radius", 4.0))
        held = {w["kind"] for w in weapons if w}
        rank = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}
        for m in obs.get("mechs") or []:
            if m.get("kind") != "mech":
                continue
            ew = m.get("weapons") or []
            if "mortar" in ew:
                rank["laser"] += 2
            if "laser" in ew:
                rank["lightning"] += 2
            if "lightning" in ew:
                rank["mortar"] += 2
        near = [
            p for p in (obs.get("pickups") or [])
            if dist2d(p["pos"], (x, y)) <= reach - 0.15
            and abs(p["pos"][2] - z) <= 3.8
            and p["kind"] not in held
            and tick - self.failed_pickups.get(int(p["id"]), -999) > 25
        ]
        if not near:
            return None, None
        near.sort(key=lambda p: (-rank.get(p["kind"], 0), dist2d(p["pos"], (x, y))))
        best = near[0]
        pid = int(best["id"])
        if weapons[1] is None:
            if best["kind"] != "gun":
                return pid, None
        elif weapons[0] is None:
            return pid, None
        else:
            w1 = weapons[1]
            if rank.get(best["kind"], 0) > rank.get(w1["kind"], 0) + 1 and float(w1.get("cooldown", 0.0)) <= 0.2:
                return pid, 1
        return None, None

    # ------------------------------------------------------------ targeting
    def _select_target(self, obs: mf.Observation) -> Tuple[Optional[Dict[str, Any]], str]:
        me = obs["self"]
        mp = me["pos"]
        mechs = obs.get("mechs") or []
        enemies = [m for m in mechs if m.get("kind") == "mech"]
        if enemies:
            enemies.sort(key=lambda m: dist3d(m["pos"], mp))
            best, best_s = None, 1e9
            for i, m in enumerate(enemies):
                d = dist3d(m["pos"], mp)
                s = d + float(m.get("hp", 700.0)) * 0.12
                if i < 4 and self._los(mp, m):
                    s -= 55.0
                if m.get("charging"):
                    s -= 40.0
                if s < best_s:
                    best, best_s = m, s
            return best, "mech"
        in_zone = me.get("in_zone", True)
        hp_ok = me["hp"] > me["max_hp"] * 0.35
        npcs = [m for m in mechs if m.get("kind") == "npc"]
        if npcs and in_zone and hp_ok:
            n = min(npcs, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(n["pos"], mp) < 65.0:
                return n, "npc"
        crates = obs.get("crates") or []
        if crates and in_zone and hp_ok:
            c = min(crates, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(c["pos"], mp) < 55.0:
                return c, "crate"
        return None, "none"

    def _aim_and_fire(self, obs, target, kind, move_dir):
        me = obs["self"]
        mx, my, mz = me["pos"]
        ayaw, apitch = me["aim"]
        mech_cfg = self.cfg.get("mech") or {}
        min_p = float(mech_cfg.get("min_pitch", -1.1))
        max_p = float(mech_cfg.get("max_pitch", 1.1))
        weapons = me.get("weapons") or [None, None]
        muz = self._muzzle
        fire = [False, False]
        reload_cmd = False

        gun = None
        for w in weapons:
            if w and w["kind"] == "gun":
                gun = w
                mag = float((w.get("stats") or {}).get("mag", 12.0))
                ammo = float(w.get("ammo", mag))
                if float(w.get("reload", 0.0)) == 0.0 and (ammo == 0 or (target is None and ammo < mag * 0.65)):
                    reload_cmd = True

        if target is None:
            tick = int(obs["tick"])
            if self.last_attacker_bearing is not None and tick - self.last_attacker_tick < 35:
                yaw = self.last_attacker_bearing
            elif self.last_sound_bearing is not None and tick - self.last_sound_tick < 25:
                yaw = self.last_sound_bearing
            elif math.hypot(move_dir[0], move_dir[1]) > 0.1:
                yaw = math.atan2(move_dir[1], move_dir[0])
            else:
                yaw = ayaw
            return [yaw, 0.0], fire, reload_cmd

        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target.get("vel") or [0.0, 0.0, 0.0]
        th = 4.5 if kind == "mech" else (1.8 if kind == "npc" else 1.0)
        hd = max(1.0, dist2d((mx, my), (tx, ty)))
        has_los = self._los(me["pos"], target, th)

        gun_speed = float(((gun or {}).get("stats") or {}).get("speed", 252.0))
        tf = clamp((hd - 3.0) / max(1.0, gun_speed), 0.0, 1.2)
        for _ in range(2):
            px, py = tx + tvx * tf, ty + tvy * tf
            pz = tz + th + tvz * tf * 0.5
            d3 = max(0.0, math.sqrt((px - mx) ** 2 + (py - my) ** 2 + (pz - mz - muz) ** 2) - 3.0)
            tf = clamp(d3 / max(1.0, gun_speed), 0.0, 1.2)
        px, py = tx + tvx * tf, ty + tvy * tf
        pz = tz + th + tvz * tf * 0.5
        gun_yaw = math.atan2(py - my, px - mx)
        gun_pitch = clamp(math.atan2(pz - mz - muz, max(1.0, math.hypot(px - mx, py - my))), min_p, max_p)
        dir_yaw = math.atan2(ty - my, tx - mx)
        dir_pitch = clamp(math.atan2(tz + th - mz - muz, hd), min_p, max_p)

        aim_yaw, aim_pitch = (gun_yaw, gun_pitch) if gun else (dir_yaw, dir_pitch)
        override: Optional[Tuple[float, float]] = None
        override_prio = 0

        # angular tolerance from target size: radius ~2.5 m, height ~9 m
        tol_y = clamp(math.atan2(2.2, hd) + 0.015, 0.035, 0.20)
        tol_p = clamp(math.atan2(3.5, hd) + 0.03, 0.06, 0.25)

        for si in (0, 1):
            w = weapons[si]
            if w is None:
                continue
            wk = w["kind"]
            st = w.get("stats") or {}
            cd = float(w.get("cooldown", 0.0))

            if wk == "gun":
                rng = float(st.get("range", 230.0))
                ammo = float(w.get("ammo", 1.0))
                if (ammo > 0 and float(w.get("reload", 0.0)) == 0.0 and hd <= rng
                        and angle_diff(ayaw, gun_yaw) < tol_y and abs(apitch - gun_pitch) < tol_p
                        and (has_los or hd < 30.0)):
                    fire[si] = True

            elif wk == "lightning":
                rng = float(st.get("range", 45.0))
                cone = math.radians(float(st.get("cone_deg", 30.0)) * 0.5)
                cost = float(st.get("energy", 3.0))
                if (cd <= 0.05 and me["energy"] >= cost + 5.0 and hd <= rng - 0.5
                        and angle_diff(ayaw, dir_yaw) <= cone * 0.8 and has_los):
                    fire[si] = True

            elif wk == "laser" and kind == "mech":
                rng = float(st.get("range", 280.0))
                full = float(st.get("charge", 0.75))
                minc = float(st.get("min_charge", 0.15))
                ch = float(w.get("charge", 0.0))
                lx, ly, lz = tx + tvx * 0.05, ty + tvy * 0.05, tz + th + tvz * 0.05
                l_yaw = math.atan2(ly - my, lx - mx)
                l_pitch = clamp(math.atan2(lz - mz - muz, max(1.0, math.hypot(lx - mx, ly - my))), min_p, max_p)
                ey, ep = angle_diff(ayaw, l_yaw), abs(apitch - l_pitch)
                tol = clamp(2.4 / max(8.0, hd), 0.03, 0.12)
                if ch > 0.0:
                    override, override_prio = (l_yaw, l_pitch), 3
                    on = ey <= tol and ep <= tol * 1.4 and has_los and hd <= rng
                    thp = float(target.get("hp", 500.0))
                    if ch >= full - 0.05 and on:
                        fire[si] = False
                    elif ch >= minc + 0.2 and on and thp <= 150.0:
                        fire[si] = False
                    elif ch >= full + 0.5:
                        fire[si] = False
                    elif not has_los and ch >= minc:
                        fire[si] = False
                    else:
                        fire[si] = True
                elif cd <= 0.05 and hd <= rng and has_los and ey < 0.45:
                    fire[si] = True
                    override, override_prio = (l_yaw, l_pitch), 3

            elif wk == "mortar" and kind in ("mech", "npc"):
                v = float(st.get("speed", 72.0))
                g = float(st.get("gravity", 22.0))
                splash = float(st.get("splash_radius", 7.0))
                gun_ready = bool(gun) and float(gun.get("ammo", 1.0)) > 0 and float(gun.get("reload", 0.0)) == 0.0
                want = (not has_los) or hd > 125.0 or not gun_ready
                if want and hd > splash + 10.0 and cd <= 0.12 and override_prio < 3:
                    te = hd / max(20.0, v * 0.78)
                    ex, ey_ = tx + tvx * te, ty + tvy * te
                    R = max(1.0, math.hypot(ex - mx, ey_ - my))
                    dh = tz - (mz + muz + 2.5)
                    v2 = v * v
                    disc = v2 * v2 - g * (g * R * R + 2.0 * dh * v2)
                    if disc >= 0.0:
                        root = math.sqrt(disc)
                        lo = math.atan2((v2 - root) / (g * R), 1.0)
                        hi = math.atan2((v2 + root) / (g * R), 1.0)
                        ch_p = lo if has_los else (hi if hi <= max_p else lo)
                        if min_p <= ch_p <= max_p:
                            m_yaw = math.atan2(ey_ - my, ex - mx)
                            override, override_prio = (m_yaw, ch_p), 1
                            if angle_diff(ayaw, m_yaw) < 0.12 and abs(apitch - ch_p) < 0.12:
                                fire[si] = True

        if override is not None:
            aim_yaw, aim_pitch = override
        return [aim_yaw, aim_pitch], fire, reload_cmd

    # ------------------------------------------------------------ zone & goals
    def _zone_now(self, obs) -> Tuple[float, float, float]:
        z = obs["zone"]
        t = float(obs["t"])
        s0, s1 = float(z["shrink_start"]), float(z["shrink_end"])
        c0, c1 = z["center"], z["next_center"]
        r0, r1 = float(z["radius"]), float(z["next_radius"])
        if t <= s0 or s1 <= s0:
            return float(c0[0]), float(c0[1]), r0
        if t >= s1:
            return float(c1[0]), float(c1[1]), r1
        a = clamp((t - s0) / (s1 - s0), 0.0, 1.0)
        return c0[0] + (c1[0] - c0[0]) * a, c0[1] + (c1[1] - c0[1]) * a, r0 + (r1 - r0) * a

    def _goal(self, obs, target, kind) -> Tuple[float, float, str, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        energy = float(me["energy"])
        zone = obs["zone"]
        nx, ny = float(zone["next_center"][0]), float(zone["next_center"][1])
        nr = float(zone["next_radius"])
        cx, cy, cr = self._zone_now(obs)
        d_cur = dist2d((mx, my), (cx, cy))
        d_next = dist2d((mx, my), (nx, ny))
        tick = int(obs["tick"])

        if not me["in_zone"] or d_cur > max(8.0, cr - 18.0):
            return nx, ny, "zone-urgent", energy > 20.0
        if float(zone["shrink_start"]) - float(obs["t"]) < 12.0 and d_next > max(12.0, nr * 0.82):
            return nx, ny, "zone-rotate", energy > 45.0

        enemies = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
        threatened = bool(enemies)

        if not threatened or hp_frac > 0.6:
            for d in obs.get("drops") or []:
                dx, dy, _ = d["pos"]
                if dist2d((dx, dy), (nx, ny)) > nr * 0.95:
                    continue
                dd = dist2d((mx, my), (dx, dy))
                li = float(d.get("lands_in", 0.0))
                holder = d.get("holder")
                if dd - li * 11.5 < 95.0 and (holder is None or holder == me["id"] or hp_frac > 0.55):
                    return dx, dy, f"drop({li:.0f}s)", (li <= 2.0 and dd > 15.0 and energy > 35.0)

        if not threatened:
            cubes = [c for c in (obs.get("cubes") or [])
                     if dist2d(c["pos"], (cx, cy)) < cr - 10.0 and abs(c["pos"][2] - mz) < 6.0]
            if cubes:
                c = min(cubes, key=lambda q: dist2d(q["pos"], (mx, my)))
                if dist2d(c["pos"], (mx, my)) < 45.0:
                    return c["pos"][0], c["pos"][1], "cube", False

        weapons = me.get("weapons") or [None, None]
        if weapons[1] is None and not threatened:
            gp = [p for p in (obs.get("pickups") or [])
                  if p["kind"] != "gun" and dist2d(p["pos"], (cx, cy)) < cr - 10.0
                  and abs(p["pos"][2] - mz) < 6.0 and tick - self.failed_pickups.get(int(p["id"]), -999) > 25]
            if gp:
                p = min(gp, key=lambda q: dist2d(q["pos"], (mx, my)))
                if dist2d(p["pos"], (mx, my)) < 55.0:
                    return p["pos"][0], p["pos"][1], f"loot-{p['kind']}", False

        if target is not None and kind == "mech":
            tx, ty, _ = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            held = {w["kind"] for w in weapons if w}
            close = [e for e in enemies if dist2d(e["pos"], (mx, my)) < 120.0]
            pref = 32.0 if "lightning" in held else (65.0 if "laser" in held else 50.0)
            if hp_frac < 0.38 or len(close) >= 3:
                pref = 100.0
            sign = 1.0 if ((tick // 18 + self.slot) % 2 == 0) else -1.0
            rx, ry = (tx - mx) / d, (ty - my) / d
            sx, sy = -ry * sign, rx * sign
            radial = clamp((d - pref) / 25.0, -0.9, 0.85)
            gx = mx + (rx * radial + sx * 0.85) * 20.0
            gy = my + (ry * radial + sy * 0.85) * 20.0
            # never kite out of the zone
            gd = dist2d((gx, gy), (cx, cy))
            if gd > cr - 14.0:
                k = max(0.0, (cr - 14.0)) / max(1.0, gd)
                gx, gy = cx + (gx - cx) * k, cy + (gy - cy) * k
            return gx, gy, "combat", bool(target.get("charging")) and energy > 30.0

        if d_next > max(18.0, nr * 0.55):
            return nx, ny, "patrol-zone", False
        if tick % 60 == 0:
            self.roam_angle = wrap_angle(self.roam_angle + self.rng.uniform(-1.2, 1.2))
        pr = min(45.0, nr * 0.4)
        return nx + math.cos(self.roam_angle) * pr, ny + math.sin(self.roam_angle) * pr, "roam", False

    def _steer(self, obs, gx, gy) -> Tuple[List[float], bool, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        energy = float(me["energy"])
        grounded = bool(me.get("grounded", True))
        cubes = float(me.get("cubes", 0.0))
        radius = 2.5 * (1.0 + 0.05 * cubes)
        height = 9.0 * (1.0 + 0.05 * cubes)

        ddx = ddy = 0.0
        for pr in obs.get("projectiles") or []:
            px, py, _ = pr["pos"]
            vx, vy, _ = pr["vel"]
            s2 = vx * vx + vy * vy
            if s2 < 1.0:
                continue
            tc = clamp(((mx - px) * vx + (my - py) * vy) / s2, 0.0, 1.2)
            d = math.hypot(mx - (px + vx * tc), my - (py + vy * tc))
            if d < (9.5 if pr.get("kind") == "mortar" else 4.5):
                pl = math.hypot(vx, vy)
                nxp, nyp = -vy / pl, vx / pl
                side = 1.0 if ((mx - px) * nxp + (my - py) * nyp) >= 0.0 else -1.0
                ddx += nxp * side * 1.5
                ddy += nyp * side * 1.5

        dx = gx - mx + ddx * 12.0
        dy = gy - my + ddy * 12.0
        dg = math.hypot(dx, dy)
        if dg < 1.0:
            return [0.0, 0.0], False, False
        want = math.atan2(dy, dx)

        moved = dist2d((mx, my), self.last_pos)
        self.last_pos = [mx, my, mz]
        if moved < 0.18 and dg > 4.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)
        if self.stuck_ticks > 6:
            want = wrap_angle(want + (1.2 if self.slot % 2 == 0 else -1.2))

        offsets = (0.0, 0.5, -0.5, 1.1, -1.1, 1.8, -1.8, 2.5, -2.5) if self.slow else \
                  (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.85, -1.85, 2.4, -2.4)
        best_yaw, best_cost = want, 1e9
        jump = jet = False
        for off in offsets:
            if abs(off) >= best_cost:
                break
            cy_ = wrap_angle(want + off)
            ux, uy = math.cos(cy_), math.sin(cy_)
            blocked = climb = False
            for sd in (4.0, 8.5, 14.0):
                tx, ty = mx + ux * sd, my + uy * sd
                blk, top = self.world.is_blocked_point(tx, ty, mz, radius, height)
                if blk:
                    if top - mz <= 16.0 and energy > 55.0 and abs(off) < 0.4:
                        climb = True
                    else:
                        blocked = True
                        break
                if self.world.terrain_z(tx, ty) - mz > sd * 1.15 + 1.2:
                    blocked = True
                    break
            if not blocked:
                cost = abs(off) + (0.6 if climb else 0.0)
                if cost < best_cost:
                    best_cost, best_yaw = cost, cy_
                    if climb:
                        jump, jet = grounded, not grounded
        if self.stuck_ticks > 4 and energy > 30.0:
            jump, jet = grounded, not grounded
        return [math.cos(best_yaw), math.sin(best_yaw)], jump, jet

    # ------------------------------------------------------------ main
    def _act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        me = obs["self"]
        mx, my, mz = me["pos"]
        self._los_cache = {}
        self._muzzle = float((me.get("stats") or {}).get("muzzle_height", 7.0))

        for ev in obs.get("events") or []:
            et = ev.get("type")
            if et == "damaged":
                if ev.get("bearing") is not None:
                    self.last_attacker_bearing = float(ev["bearing"])
                    self.last_attacker_tick = tick
                self.last_dmg = (str(ev.get("weapon")), float(ev.get("amount", 0.0)), bool(ev.get("rear")))
            elif et == "take_failed":
                self.failed_pickups[int(ev.get("id", -1))] = tick
        near_s = [s for s in (obs.get("sounds") or []) if s.get("dist") in ("near", "mid")]
        if near_s:
            sb = min(near_s, key=lambda s: float(s.get("age", 1.0)))
            self.last_sound_bearing = float(sb["bearing"])
            self.last_sound_tick = tick

        target, kind = self._select_target(obs)
        gx, gy, self.mode, boost_goal = self._goal(obs, target, kind)
        move_dir, jump_nav, jet_nav = self._steer(obs, gx, gy)
        aim, fire, reload_cmd = self._aim_and_fire(obs, target, kind, move_dir)

        grounded = bool(me.get("grounded", True))
        act: Dict[str, Any] = {
            "tick": obs["tick"],
            "move": [round(move_dir[0], 4), round(move_dir[1], 4)],
            "aim": [round(aim[0], 4), round(aim[1], 4)],
            "fire": [bool(fire[0]), bool(fire[1])],
            "boost": bool((boost_goal and grounded) or (jet_nav and not grounded)),
            "jump": bool(jump_nav),
            "reload": bool(reload_cmd),
        }
        if obs.get("offer") is not None:
            act["pick"] = self._pick_upgrade(obs)
        take_id, drop_slot = self._evaluate_loot(obs)
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        mf.draw.label(f"{self.mode} hp:{me['hp']:.0f} e:{me['energy']:.0f}")
        return act

    def _fallback(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        z = obs["zone"]
        dx, dy = z["next_center"][0] - me["pos"][0], z["next_center"][1] - me["pos"][1]
        d = max(1.0, math.hypot(dx, dy))
        return {"tick": obs["tick"], "move": [dx / d, dy / d], "fire": [False, False]}

    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        tick = int(obs["tick"])
        self.slow = tick < self.slow_until
        try:
            a = self._act(obs)
        except Exception:
            self._errors += 1
            if self._errors <= 3:
                mf.log(traceback.format_exc())
            a = self._fallback(obs)
        dt = time.perf_counter() - t0
        self.times.append(dt)
        if dt > self.budget * 0.6:
            self.slow_until = tick + 20
        if dt > self.budget * 0.9:
            self.late += 1
        if tick % 100 == 0 and self.times:
            ts = sorted(self.times)
            me = obs["self"]
            mf.log(
                f"[t={obs['t']:.0f}s alive={obs['alive']} hp={me['hp']:.0f}/{me['max_hp']:.0f} "
                f"mode={self.mode} w={'/'.join(w['kind'] if w else '-' for w in me['weapons'])} "
                f"lastdmg={self.last_dmg} act_ms p50={ts[len(ts) // 2] * 1000:.1f} "
                f"p99={ts[int(len(ts) * 0.99)] * 1000:.1f} max={ts[-1] * 1000:.1f} near_budget={self.late}]"
            )
            self.times = []
        return a


if __name__ == "__main__":
    mf.run(Bot)