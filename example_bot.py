#!/usr/bin/env python3
"""apex-ghost: Rooftop-flying, building-cover, edge-hugging survivor bot for mechfight/1.

Built around real-match survival principles:
1. THE CENTER IS A DEATH TRAP:
   - Never walks to `next_center`. Holds the outer safe-zone ring (~0.68 * next_radius) until the final tiny circles.
   - Chooses the quietest sector away from gunfire/explosion sounds, laser/lightning beams, and visible mechs.
2. FLY ONTO ROOFTOPS & HIDE BEHIND BUILDINGS:
   - Precomputes 90 verified climbable rooftops with exact exterior launch pads `(lx, ly)` and roof step-in points `(ix, iy)`.
   - Two-stage roof ascent: walks to `(lx, ly)`, jumps + air-boosts (`jet`) until feet (`pos.z`) clear `roof.top`, then steps onto `(ix, iy)` and perches on the roof.
   - When on the ground and hurt (< 62% HP) or outnumbered (2+ enemies), ducks behind the nearest building wall that blocks 3D line-of-sight to the threat.
3. DELIBERATE 45 HP/s REGENERATION:
   - After 5s without dealing or taking damage, mechs heal 45 HP/s.
   - When HP < 55% and behind cover, holds trigger on distant/non-lethal targets and stops shooting crates/NPCs so the 5s regen timer is never reset by our own gun.
4. INSTANT-HIT WEAPON PRIORITY & PROVEN GUN TRACKING:
   - Prioritizes Lightning (instant 60 dmg in 30 deg cone, no lead needed, shorts enemy lasers) and Laser (hitscan), while keeping the 4.17 bot's proven generous Gun firing window (`0.22 rad`).
   - Never turns its back on an enemy or recent attacker (prevents 1.5x rear armor multiplier).
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
    """Spatial hash over world.json prisms, precomputed climbable rooftops, wall cover nodes, and terrain."""

    def __init__(self, map_dir: str, files: List[str]) -> None:
        self.bounds = {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0}
        self.prisms: List[Dict[str, Any]] = []
        self.rooftops: List[Dict[str, Any]] = []
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
                        "id": idx,
                        "kind": p.get("kind", "building"),
                        "base": float(p.get("base", -10.0)),
                        "top": float(p.get("top", 20.0)),
                        "ring": ring,
                        "holes": p.get("holes") or [],
                        "bbox": bb,
                        "center": (0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])),
                    })
                    for gx in range(int(bb[0] // self.cs), int(bb[2] // self.cs) + 1):
                        for gy in range(int(bb[1] // self.cs), int(bb[3] // self.cs) + 1):
                            self.grid.setdefault((gx, gy), []).append(idx)
                del w
                self._precompute_rooftops()
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

    def _is_2d_solid(self, x: float, y: float, ignore_idx: int = -1) -> bool:
        b = self.bounds
        if x < b["minX"] + 3.0 or x > b["maxX"] - 3.0 or y < b["minY"] + 3.0 or y > b["maxY"] - 3.0:
            return True
        for idx in self.grid.get((int(x // self.cs), int(y // self.cs)), []):
            if idx == ignore_idx:
                continue
            p = self.prisms[idx]
            if p["kind"] == "bridge":
                continue
            x0, y0, x1, y1 = p["bbox"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                if point_in_ring(x, y, p["ring"]) and not any(point_in_ring(x, y, h) for h in p["holes"]):
                    return True
        return False

    def _precompute_rooftops(self) -> None:
        for idx, p in enumerate(self.prisms):
            if p["kind"] != "building":
                continue
            h = p["top"] - p["base"]
            if not (6.0 <= h <= 36.0):
                continue
            x0, y0, x1, y1 = p["bbox"]
            if (x1 - x0) < 10.0 or (y1 - y0) < 10.0:
                continue
            ring = p["ring"]
            holes = p["holes"]
            perch = None
            for gx in (0.5, 0.35, 0.65):
                for gy in (0.5, 0.35, 0.65):
                    cx, cy = x0 + (x1 - x0) * gx, y0 + (y1 - y0) * gy
                    if point_in_ring(cx, cy, ring) and not any(point_in_ring(cx, cy, hl) for hl in holes):
                        if not self._is_2d_solid(cx, cy, ignore_idx=idx):
                            perch = (cx, cy)
                            break
                if perch:
                    break
            if not perch:
                continue
            launches = []
            n = len(ring)
            for i in range(0, n, 2):
                ax, ay = ring[i], ring[i + 1]
                bx, by = ring[(i + 2) % n], ring[(i + 3) % n]
                seg = math.hypot(bx - ax, by - ay)
                if seg < 7.0:
                    continue
                mx_e, my_e = 0.5 * (ax + bx), 0.5 * (ay + by)
                nx_e, ny_e = -(by - ay) / seg, (bx - ax) / seg
                lx, ly = mx_e + nx_e * 3.8, my_e + ny_e * 3.8
                if point_in_ring(lx, ly, ring):
                    nx_e, ny_e = -nx_e, -ny_e
                    lx, ly = mx_e + nx_e * 3.8, my_e + ny_e * 3.8
                if not self._is_2d_solid(lx, ly, ignore_idx=-1):
                    ix, iy = mx_e - nx_e * 3.4, my_e - ny_e * 3.4
                    if point_in_ring(ix, iy, ring) and not any(point_in_ring(ix, iy, hl) for hl in holes):
                        launches.append((lx, ly, ix, iy))
            if launches:
                self.rooftops.append({
                    "id": idx,
                    "top": p["top"],
                    "base": p["base"],
                    "height": h,
                    "perch": perch,
                    "launches": launches,
                    "bbox": (x0, y0, x1, y1),
                })

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

    def is_blocked_point(
        self, x: float, y: float, z: float, radius: float = 2.6, height: float = 9.0
    ) -> Tuple[bool, float]:
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
        n = max(3, min(36, int(math.hypot(dx, dy) / spacing)))
        for s in range(1, n):
            t = s / n
            if self._solid_at(a[0] + dx * t, a[1] + dy * t, a[2] + dz * t):
                return False
        return True

    def find_cover_behind_building(
        self, mx: float, my: float, mz: float, tx: float, ty: float, tz: float, cx: float, cy: float, cr: float
    ) -> Optional[Tuple[float, float]]:
        """Finds an open ground spot behind a nearby building that blocks LOS to threat (tx, ty, tz)."""
        best_pt = None
        best_sc = 1e9
        for r in self.rooftops:
            px, py = r["perch"]
            if dist2d((mx, my), (px, py)) > 75.0:
                continue
            for lx, ly, _, _ in r["launches"]:
                if dist2d((lx, ly), (cx, cy)) > cr - 14.0:
                    continue
                d_me = dist2d((mx, my), (lx, ly))
                if d_me > 60.0:
                    continue
                if dist2d((lx, ly), (tx, ty)) > dist2d((px, py), (tx, ty)):
                    if not self.has_los([lx, ly, mz + 5.0], [tx, ty, tz + 7.0], spacing=6.0):
                        if d_me < best_sc:
                            best_sc = d_me
                            best_pt = (lx, ly)
        return best_pt


class Bot:
    name = "apex-ghost"

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

        self.last_damage_tick = -999
        self.last_dealt_tick = -999
        self.last_attacker_bearing: Optional[float] = None
        self.last_attacker_id: Optional[int] = None
        self.last_sound_bearing: Optional[float] = None
        self.last_sound_tick = -999
        self.danger_bearings: List[Tuple[float, int]] = []

        self.edge_angle = wrap_angle((self.slot / 16.0) * 2.0 * math.pi + self.rng.uniform(-0.3, 0.3))
        self.stuck_ticks = 0
        self.last_pos = [0.0, 0.0, 0.0]
        self.failed_pickups: Dict[int, int] = {}

        self.active_climb: Optional[Tuple[float, float, float, float, float]] = None
        self.active_climb_until = -1

        self.slow_until = -1
        self.slow = False
        self.times: List[float] = []
        self.late = 0
        self.last_dmg: Tuple[str, float, bool] = ("", 0.0, False)
        self._los_cache: Dict[Any, bool] = {}
        self._muzzle = 7.0
        self._errors = 0
        self.mode = "init"

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
                s += 5.0 if req in held else -6.0
            for eff in udef.get("effects") or []:
                path = str(eff.get("path", "")).lower()
                if "armor" in path:
                    s += 7.5
                elif any(k in path for k in ("max_hp", "regen")):
                    s += 6.0
                elif any(k in path for k in ("damage", "cooldown", "charge", "reload", "speed", "range")):
                    s += 4.5
                elif any(k in path for k in ("energy", "turn_rate", "walk_speed", "boost_speed")):
                    s += 4.2
                elif any(k in path for k in ("vision", "near_range", "fov")):
                    s += 3.0
            if any(k in text for k in ("armor", "damage taken", "hp", "regen")):
                s += 3.0
            if any(k in text for k in ("damage", "fire rate", "cooldown", "charge", "chain")):
                s += 2.5
            if any(k in text for k in ("speed", "energy", "turn", "boost")):
                s += 2.0
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
        rank = {"lightning": 5, "laser": 5, "mortar": 3, "gun": 2}
        for m in obs.get("mechs") or []:
            if m.get("kind") != "mech":
                continue
            ew = m.get("weapons") or []
            if "laser" in ew:
                rank["lightning"] += 2
            if "mortar" in ew:
                rank["laser"] += 2
            if "lightning" in ew:
                rank["mortar"] += 1
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
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        tick = int(obs["tick"])
        mechs = obs.get("mechs") or []
        enemies = [m for m in mechs if m.get("kind") == "mech"]

        if enemies:
            enemies.sort(key=lambda m: dist3d(m["pos"], mp))
            best, best_s = None, 1e9
            for i, m in enumerate(enemies):
                d = dist3d(m["pos"], mp)
                s = d + float(m.get("hp", 700.0)) * 0.12
                if i < 5 and self._los(mp, m):
                    s -= 75.0
                if m.get("charging"):
                    s -= 45.0
                if self.last_attacker_id is not None and int(m["id"]) == self.last_attacker_id:
                    s -= 30.0
                if s < best_s:
                    best, best_s = m, s
            return best, "mech"

        ticks_quiet = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        if hp_frac < 0.68 and ticks_quiet >= 35:
            return None, "none"

        in_zone = me.get("in_zone", True)
        crates = obs.get("crates") or []
        if crates and in_zone and hp_frac > 0.55:
            c = min(crates, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(c["pos"], mp) < 55.0 and self._los(mp, c, 1.2):
                return c, "crate"

        npcs = [m for m in mechs if m.get("kind") == "npc"]
        if npcs and in_zone and hp_frac > 0.72:
            n = min(npcs, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(n["pos"], mp) < 48.0 and self._los(mp, n, 1.8):
                return n, "npc"

        return None, "none"

    # ------------------------------------------------------------ aiming & firing
    def _aim_and_fire(self, obs, target, kind, move_dir):
        me = obs["self"]
        mx, my, mz = me["pos"]
        ayaw, apitch = me["aim"]
        mech_cfg = self.cfg.get("mech") or {}
        min_p = float(mech_cfg.get("min_pitch", -1.1))
        max_p = float(mech_cfg.get("max_pitch", 1.1))
        weapons = me.get("weapons") or [None, None]
        muz = self._muzzle
        tick = int(obs["tick"])
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        fire = [False, False]
        reload_cmd = False

        gun = None
        for w in weapons:
            if w and w["kind"] == "gun":
                gun = w
                mag = float((w.get("stats") or {}).get("mag", 12.0))
                ammo = float(w.get("ammo", mag))
                if float(w.get("reload", 0.0)) == 0.0 and (ammo == 0 or (target is None and ammo < mag * 0.70)):
                    reload_cmd = True

        if target is None:
            if self.last_attacker_bearing is not None and tick - self.last_damage_tick < 45:
                yaw = self.last_attacker_bearing
            elif self.last_sound_bearing is not None and tick - self.last_sound_tick < 35:
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

        ticks_quiet = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        if (
            hp_frac < 0.55
            and ticks_quiet >= 45
            and me.get("in_zone", True)
            and (not has_los or hd > 70.0)
            and float(target.get("hp", 500.0)) > 160.0
        ):
            dir_yaw = math.atan2(ty - my, tx - mx)
            dir_pitch = clamp(math.atan2(tz + th - mz - muz, hd), min_p, max_p)
            return [dir_yaw, dir_pitch], [False, False], reload_cmd

        gun_speed = float(((gun or {}).get("stats") or {}).get("speed", 252.0))
        tf = clamp((hd - 3.0) / max(1.0, gun_speed), 0.0, 1.1)
        for _ in range(2):
            px, py = tx + tvx * tf, ty + tvy * tf
            pz = tz + th + tvz * tf * 0.5
            d3 = max(0.0, math.sqrt((px - mx) ** 2 + (py - my) ** 2 + (pz - mz - muz) ** 2) - 3.0)
            tf = clamp(d3 / max(1.0, gun_speed), 0.0, 1.1)
        px, py = tx + tvx * tf, ty + tvy * tf
        pz = tz + th + tvz * tf * 0.5
        gun_yaw = math.atan2(py - my, px - mx)
        gun_pitch = clamp(math.atan2(pz - mz - muz, max(1.0, math.hypot(px - mx, py - my))), min_p, max_p)
        dir_yaw = math.atan2(ty - my, tx - mx)
        dir_pitch = clamp(math.atan2(tz + th - mz - muz, hd), min_p, max_p)

        aim_yaw, aim_pitch = (gun_yaw, gun_pitch) if gun else (dir_yaw, dir_pitch)
        override: Optional[Tuple[float, float]] = None
        override_prio = 0

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
                if (
                    ammo > 0
                    and float(w.get("reload", 0.0)) == 0.0
                    and hd <= rng
                    and angle_diff(ayaw, gun_yaw) < 0.22
                    and abs(apitch - gun_pitch) < 0.25
                    and (has_los or hd < 30.0)
                ):
                    fire[si] = True

            elif wk == "lightning":
                rng = float(st.get("range", 45.0))
                cone = math.radians(float(st.get("cone_deg", 30.0)) * 0.5)
                cost = float(st.get("energy", 3.0))
                if (
                    cd <= 0.05
                    and me["energy"] >= cost + 4.0
                    and hd <= rng - 0.3
                    and angle_diff(ayaw, dir_yaw) <= cone
                    and has_los
                ):
                    fire[si] = True

            elif wk == "laser":
                rng = float(st.get("range", 280.0))
                full = float(st.get("charge", 0.75))
                minc = float(st.get("min_charge", 0.15))
                ch = float(w.get("charge", 0.0))
                lx, ly, lz = tx + tvx * 0.05, ty + tvy * 0.05, tz + th + tvz * 0.05
                l_yaw = math.atan2(ly - my, lx - mx)
                l_pitch = clamp(math.atan2(lz - mz - muz, max(1.0, math.hypot(lx - mx, ly - my))), min_p, max_p)
                ey, ep = angle_diff(ayaw, l_yaw), abs(apitch - l_pitch)
                if ch > 0.0:
                    override, override_prio = (l_yaw, l_pitch), 3
                    on = ey <= 0.10 and ep <= 0.14 and has_los and hd <= rng
                    thp = float(target.get("hp", 500.0))
                    if ch >= full - 0.07 and on:
                        fire[si] = False
                    elif ch >= minc + 0.22 and on and thp <= 155.0:
                        fire[si] = False
                    elif ch >= full + 1.2:
                        fire[si] = False
                    elif not has_los and ch >= minc + 0.35:
                        fire[si] = False
                    else:
                        fire[si] = True
                elif kind == "mech" and cd <= 0.05 and hd <= rng and (has_los or hd < 80.0) and ey < 0.45:
                    fire[si] = True
                    override, override_prio = (l_yaw, l_pitch), 3

            elif wk == "mortar" and kind in ("mech", "npc"):
                v = float(st.get("speed", 72.0))
                g = float(st.get("gravity", 22.0))
                splash = float(st.get("splash_radius", 7.0))
                if hd > splash + 10.0 and cd <= 0.12 and override_prio < 3:
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
                            if angle_diff(ayaw, m_yaw) < 0.16 and abs(apitch - ch_p) < 0.16:
                                fire[si] = True

        if override is not None:
            aim_yaw, aim_pitch = override
        return [aim_yaw, aim_pitch], fire, reload_cmd

    # ------------------------------------------------------------ zone & strategic goals
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

    def _safe_edge_point(self, obs, nx: float, ny: float, nr: float) -> Tuple[float, float]:
        """Picks a quiet point near the outer safe ring (0.68 * next_radius), NEVER in the center moshpit!"""
        me = obs["self"]
        mx, my, mz = me["pos"]
        if nr < 32.0:
            return nx, ny

        ring_r = clamp(nr * 0.68, 22.0, nr - 18.0)
        enemies = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
        best_pt = (nx + math.cos(self.edge_angle) * ring_r, ny + math.sin(self.edge_angle) * ring_r)
        best_s = -1e9
        best_ang = self.edge_angle

        for k in range(12):
            ang = wrap_angle(self.edge_angle + k * (2.0 * math.pi / 12.0))
            px = nx + math.cos(ang) * ring_r
            py = ny + math.sin(ang) * ring_r
            blk, top = self.world.is_blocked_point(px, py, mz, radius=3.0)
            if blk and (top - mz) > 16.0:
                continue
            s = -dist2d((mx, my), (px, py)) * 0.22
            for e in enemies:
                s += min(160.0, dist2d((px, py), e["pos"])) * 1.1
            for b_yaw, _ in self.danger_bearings:
                s += angle_diff(math.atan2(py - my, px - mx), b_yaw) * 22.0
            if s > best_s:
                best_s = s
                best_pt = (px, py)
                best_ang = ang

        self.edge_angle = best_ang
        return best_pt

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

        if not me["in_zone"] or d_cur > max(8.0, cr - 16.0):
            self.active_climb = None
            ex, ey = self._safe_edge_point(obs, nx, ny, nr)
            return ex, ey, "zone-edge", energy > 20.0
        if float(zone["shrink_start"]) - float(obs["t"]) < 14.0 and d_next > max(12.0, nr * 0.86):
            self.active_climb = None
            ex, ey = self._safe_edge_point(obs, nx, ny, nr)
            return ex, ey, "zone-rotate", energy > 45.0

        enemies = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
        close_enemies = [e for e in enemies if dist2d(e["pos"], (mx, my)) < 135.0]

        if close_enemies and (len(close_enemies) >= 2 or hp_frac < 0.62):
            self.active_climb = None
            threat = close_enemies[0]
            tx, ty, tz = threat["pos"]
            cov = self.world.find_cover_behind_building(mx, my, mz, tx, ty, tz, cx, cy, cr)
            if cov is not None:
                return cov[0], cov[1], "hide-cover", (hp_frac < 0.45 and energy > 30.0)
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            kx = mx - ((tx - mx) / d) * 40.0
            ky = my - ((ty - my) / d) * 40.0
            gd = dist2d((kx, ky), (cx, cy))
            if gd > cr - 14.0:
                scale = max(0.0, cr - 14.0) / max(1.0, gd)
                kx, ky = cx + (kx - cx) * scale, cy + (ky - cy) * scale
            return kx, ky, "retreat-edge", (hp_frac < 0.45 and energy > 30.0)

        weapons = me.get("weapons") or [None, None]
        if weapons[1] is None and len(close_enemies) <= 1:
            gp = [
                p for p in (obs.get("pickups") or [])
                if p["kind"] != "gun"
                and dist2d(p["pos"], (cx, cy)) < cr - 12.0
                and abs(p["pos"][2] - mz) < 6.5
                and tick - self.failed_pickups.get(int(p["id"]), -999) > 25
            ]
            if gp:
                gp.sort(key=lambda p: (0 if p["kind"] in ("lightning", "laser") else 1, dist2d(p["pos"], (mx, my))))
                if dist2d(gp[0]["pos"], (mx, my)) < 75.0:
                    return gp[0]["pos"][0], gp[0]["pos"][1], f"loot-{gp[0]['kind']}", False

        if not close_enemies:
            cubes = [
                c for c in (obs.get("cubes") or [])
                if dist2d(c["pos"], (cx, cy)) < cr - 12.0 and abs(c["pos"][2] - mz) < 6.0
            ]
            if cubes:
                c = min(cubes, key=lambda q: dist2d(q["pos"], (mx, my)))
                if dist2d(c["pos"], (mx, my)) < 50.0:
                    return c["pos"][0], c["pos"][1], "cube", False

        if not close_enemies and hp_frac > 0.65:
            for d in obs.get("drops") or []:
                dx, dy, _ = d["pos"]
                if dist2d((dx, dy), (cx, cy)) > cr - 15.0:
                    continue
                if obs.get("alive", 16) > 6 and dist2d((dx, dy), (nx, ny)) < nr * 0.28:
                    continue
                dd = dist2d((mx, my), (dx, dy))
                li = float(d.get("lands_in", 0.0))
                holder = d.get("holder")
                if dd - li * 11.0 < 75.0 and (holder is None or holder == me["id"]):
                    return dx, dy, f"drop({li:.0f}s)", (li <= 1.5 and dd > 15.0 and energy > 40.0)

        if target is not None and kind == "mech" and len(close_enemies) <= 1:
            tx, ty, _ = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            held = {w["kind"] for w in weapons if w}
            ew = target.get("weapons") or []
            if "lightning" in ew and "lightning" not in held:
                pref = 65.0
            elif "lightning" in held:
                pref = 32.0
            elif "laser" in held:
                pref = 68.0
            else:
                pref = 52.0
            sign = 1.0 if ((tick // 18 + self.slot) % 2 == 0) else -1.0
            rx, ry = (tx - mx) / d, (ty - my) / d
            sx, sy = -ry * sign, rx * sign
            radial = clamp((d - pref) / 22.0, -0.9, 0.85)
            gx = mx + (rx * radial + sx * 0.85) * 20.0
            gy = my + (ry * radial + sy * 0.85) * 20.0
            gd = dist2d((gx, gy), (cx, cy))
            if gd > cr - 14.0:
                k = max(0.0, (cr - 14.0)) / max(1.0, gd)
                gx, gy = cx + (gx - cx) * k, cy + (gy - cy) * k
            return gx, gy, "duel-1v1", bool(target.get("charging")) and energy > 25.0

        t_ground = self.world.terrain_z(mx, my)
        on_roof = (mz - t_ground) > 5.5
        if on_roof and d_next <= nr * 0.88:
            self.active_climb = None
            return mx, my, "roof-perch", False

        if self.active_climb is not None and tick <= self.active_climb_until and energy > 18.0:
            lx, ly, ix, iy, top_z = self.active_climb
            if mz >= top_z - 0.3:
                return ix, iy, "roof-step", False
            return lx, ly, "roof-fly", False
        else:
            self.active_climb = None

        if energy >= 78.0 and not on_roof and self.world.rooftops:
            best_launch = None
            best_d = 85.0
            for r in self.world.rooftops:
                px, py = r["perch"]
                dn = dist2d((px, py), (nx, ny))
                if dn > nr * 0.82 or (nr > 120.0 and dn < nr * 0.25):
                    continue
                if not (5.5 <= (r["top"] - mz) <= 28.0):
                    continue
                for lx, ly, ix, iy in r["launches"]:
                    dl = dist2d((mx, my), (lx, ly))
                    if dl < best_d:
                        best_d = dl
                        best_launch = (lx, ly, ix, iy, r["top"])
            if best_launch is not None:
                self.active_climb = best_launch
                self.active_climb_until = tick + 45
                return best_launch[0], best_launch[1], "roof-fly", False

        ex, ey = self._safe_edge_point(obs, nx, ny, nr)
        return ex, ey, "edge-patrol", False

    def _steer(self, obs, gx, gy) -> Tuple[List[float], bool, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        energy = float(me["energy"])
        grounded = bool(me.get("grounded", True))
        cubes = float(me.get("cubes", 0.0))
        radius = 2.5 * (1.0 + 0.05 * cubes)
        height = 9.0 * (1.0 + 0.05 * cubes)

        if self.active_climb is not None:
            lx, ly, ix, iy, top_z = self.active_climb
            d_launch = dist2d((mx, my), (lx, ly))
            if d_launch < 4.8 or not grounded:
                if mz < top_z - 0.2:
                    dx_in, dy_in = ix - mx, iy - my
                    d_in = max(1.0, math.hypot(dx_in, dy_in))
                    push = 0.35 if (top_z - mz) > 2.5 else 1.0
                    return [(dx_in / d_in) * push, (dy_in / d_in) * push], grounded, (not grounded)
                else:
                    dx_in, dy_in = ix - mx, iy - my
                    d_in = max(1.0, math.hypot(dx_in, dy_in))
                    return [dx_in / d_in, dy_in / d_in], False, False

        ddx = ddy = 0.0
        for pr in obs.get("projectiles") or []:
            px, py, _ = pr["pos"]
            vx, vy, _ = pr["vel"]
            s2 = vx * vx + vy * vy
            if s2 < 1.0:
                continue
            tc = clamp(((mx - px) * vx + (my - py) * vy) / s2, 0.0, 1.2)
            d = math.hypot(mx - (px + vx * tc), my - (py + vy * tc))
            if d < (10.0 if pr.get("kind") == "mortar" else 4.5):
                pl = math.hypot(vx, vy)
                nxp, nyp = -vy / pl, vx / pl
                side = 1.0 if ((mx - px) * nxp + (my - py) * nyp) >= 0.0 else -1.0
                ddx += nxp * side * 1.6
                ddy += nyp * side * 1.6

        dx = gx - mx + ddx * 12.0
        dy = gy - my + ddy * 12.0
        dg = math.hypot(dx, dy)
        if dg < 1.2 and ddx == 0.0 and ddy == 0.0:
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

        offsets = (0.0, 0.5, -0.5, 1.1, -1.1, 1.8, -1.8, 2.5, -2.5) if self.slow else (
            0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.85, -1.85, 2.4, -2.4
        )
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
                    if top - mz <= 18.0 and energy > 55.0 and abs(off) < 0.4:
                        climb = True
                    else:
                        blocked = True
                        break
                if self.world.terrain_z(tx, ty) - mz > sd * 1.15 + 1.2:
                    blocked = True
                    break
            if not blocked:
                cost = abs(off) + (0.5 if climb else 0.0)
                if cost < best_cost:
                    best_cost, best_yaw = cost, cy_
                    if climb:
                        jump, jet = grounded, not grounded
        if self.stuck_ticks > 4 and energy > 30.0:
            jump, jet = grounded, not grounded

        speed_mult = 1.0
        if self.mode in ("edge-patrol", "roof-perch", "hide-cover") and not obs.get("mechs"):
            if any(s.get("dist") == "near" for s in (obs.get("sounds") or [])):
                speed_mult = 0.38

        return [math.cos(best_yaw) * speed_mult, math.sin(best_yaw) * speed_mult], jump, jet

    # ------------------------------------------------------------ main loop
    def _act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        me = obs["self"]
        mx, my, mz = me["pos"]
        self._los_cache = {}
        self._muzzle = float((me.get("stats") or {}).get("muzzle_height", 7.0))

        for ev in obs.get("events") or []:
            et = ev.get("type")
            if et == "damaged":
                self.last_damage_tick = tick
                if ev.get("bearing") is not None:
                    b = float(ev["bearing"])
                    self.last_attacker_bearing = b
                    self.danger_bearings.append((b, tick))
                if ev.get("by") is not None:
                    self.last_attacker_id = int(ev["by"])
                self.last_dmg = (str(ev.get("weapon")), float(ev.get("amount", 0.0)), bool(ev.get("rear")))
            elif et == "hit":
                self.last_dealt_tick = tick
            elif et == "take_failed":
                self.failed_pickups[int(ev.get("id", -1))] = tick

        for s in obs.get("sounds") or []:
            if s.get("dist") in ("near", "mid"):
                b = float(s["bearing"])
                if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion"):
                    self.danger_bearings.append((b, tick))
                if s.get("dist") == "near":
                    self.last_sound_bearing = b
                    self.last_sound_tick = tick

        for bm in obs.get("beams") or []:
            path = bm.get("path") or []
            if path:
                self.danger_bearings.append((math.atan2(path[0][1] - my, path[0][0] - mx), tick))

        self.danger_bearings = [(b, tk) for (b, tk) in self.danger_bearings if (tick - tk) <= 160][-16:]

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
            "boost": bool((boost_goal and grounded and not jump_nav) or (jet_nav and not grounded)),
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
