#!/usr/bin/env python3
"""apex-stakeout: Precision Stakeout & Ambush Bot for mechfight/1."""
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
    """Spatial hash over world.json prisms, precomputed building cover posts, and terrain heightfield."""

    def __init__(self, map_dir: str, files: List[str]) -> None:
        self.bounds = {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0}
        self.prisms: List[Dict[str, Any]] = []
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        self.cover_posts: List[Tuple[float, float, float, float]] = []
        self.cover_grid: Dict[Tuple[int, int], List[int]] = {}
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
                        "center": (0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])),
                    })
                    for gx in range(int(bb[0] // self.cs), int(bb[2] // self.cs) + 1):
                        for gy in range(int(bb[1] // self.cs), int(bb[3] // self.cs) + 1):
                            self.grid.setdefault((gx, gy), []).append(idx)
                del w
                self._precompute_cover_posts()
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

    def _is_2d_solid(self, x: float, y: float) -> bool:
        b = self.bounds
        if x < b["minX"] + 3.0 or x > b["maxX"] - 3.0 or y < b["minY"] + 3.0 or y > b["maxY"] - 3.0:
            return True
        for idx in self.grid.get((int(x // self.cs), int(y // self.cs)), []):
            p = self.prisms[idx]
            if p["kind"] == "bridge":
                continue
            x0, y0, x1, y1 = p["bbox"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                if point_in_ring(x, y, p["ring"]) and not any(point_in_ring(x, y, h) for h in p["holes"]):
                    return True
        return False

    def _precompute_cover_posts(self) -> None:
        t_end = time.perf_counter() + 5.0  # never risk the 20 s ready budget
        for p in self.prisms:
            if time.perf_counter() > t_end:
                mf.log("[map] cover post precompute cut at 5 s")
                break
            if p["kind"] == "bridge" or (p["top"] - p["base"]) < 5.5:
                continue
            ring = p["ring"]
            n = len(ring)
            pcx, pcy = p["center"]
            for i in range(0, n, 2):
                ax, ay = ring[i], ring[i + 1]
                bx, by = ring[(i + 2) % n], ring[(i + 3) % n]
                seg = math.hypot(bx - ax, by - ay)
                if seg < 6.5:
                    continue
                mx_e, my_e = 0.5 * (ax + bx), 0.5 * (ay + by)
                nx_e, ny_e = -(by - ay) / seg, (bx - ax) / seg
                wx, wy = mx_e + nx_e * 3.8, my_e + ny_e * 3.8
                if point_in_ring(wx, wy, ring):
                    nx_e, ny_e = -nx_e, -ny_e
                    wx, wy = mx_e + nx_e * 3.8, my_e + ny_e * 3.8
                if not self._is_2d_solid(wx, wy):
                    if not any(
                        self._is_2d_solid(wx + dx, wy + dy)
                        for dx, dy in ((2.6, 0.0), (-2.6, 0.0), (0.0, 2.6), (0.0, -2.6))
                    ):
                        c_idx = len(self.cover_posts)
                        self.cover_posts.append((wx, wy, pcx, pcy))
                        cg = (int(wx // self.cs), int(wy // self.cs))
                        self.cover_grid.setdefault(cg, []).append(c_idx)

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

    def has_los(self, a, b, spacing: float = 4.5) -> bool:
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = max(3, min(36, int(math.hypot(dx, dy) / spacing)))
        for s in range(1, n):
            t = s / n
            if self._solid_at(a[0] + dx * t, a[1] + dy * t, a[2] + dz * t):
                return False
        return True

    def find_combat_cover(
        self, mx: float, my: float, mz: float, tx: float, ty: float, tz: float,
        cx: float, cy: float, cr: float, want_los: bool
    ) -> Optional[Tuple[float, float]]:
        gx, gy = int(mx // self.cs), int(my // self.cs)
        best_pt = None
        best_sc = 1e9
        for dx_g in (-2, -1, 0, 1, 2):
            for dy_g in (-2, -1, 0, 1, 2):
                for c_idx in self.cover_grid.get((gx + dx_g, gy + dy_g), []):
                    wx, wy, pcx, pcy = self.cover_posts[c_idx]
                    d_me = dist2d((mx, my), (wx, wy))
                    if d_me > 60.0 or dist2d((wx, wy), (cx, cy)) > cr - 14.0:
                        continue
                    d_enemy = dist2d((wx, wy), (tx, ty))
                    if not want_los:
                        if d_enemy > dist2d((pcx, pcy), (tx, ty)) and d_enemy > 18.0:
                            if not self.has_los([wx, wy, mz + 6.5], [tx, ty, tz + 6.5], spacing=6.0):
                                sc = d_me - min(40.0, d_enemy * 0.25)
                                if sc < best_sc:
                                    best_sc = sc
                                    best_pt = (wx, wy)
                    else:
                        if d_me < 24.0 and 28.0 <= d_enemy <= 90.0:
                            if self.has_los([wx, wy, mz + 7.0], [tx, ty, tz + 4.5], spacing=5.5):
                                sc = d_me + abs(d_enemy - 52.0) * 0.35
                                if sc < best_sc:
                                    best_sc = sc
                                    best_pt = (wx, wy)
        return best_pt


class Bot:
    name = "apex-stakeout"

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
        self.last_sound_tick: int = -999

        self.roam_angle = wrap_angle((self.slot / 16.0) * 2.0 * math.pi + self.rng.uniform(-0.4, 0.4))
        self.stakeout_pt: Optional[Tuple[float, float]] = None
        self.stakeout_until = -1
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

    def _los(self, me_pos, m: Dict[str, Any], h: float = 4.5) -> bool:
        key = m.get("id")
        r = self._los_cache.get(key)
        if r is None:
            p = m["pos"]
            r = self.world.has_los(
                [me_pos[0], me_pos[1], me_pos[2] + self._muzzle],
                [p[0], p[1], p[2] + h],
                8.5 if self.slow else 4.5,
            )
            self._los_cache[key] = r
        return r

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
                s += 4.8 if req in held else -5.0
            for eff in udef.get("effects") or []:
                path = str(eff.get("path", "")).lower()
                if any(k in path for k in ("armor", "max_hp", "regen")):
                    s += 5.5
                elif any(k in path for k in ("damage", "cooldown", "charge", "reload", "speed", "range")):
                    s += 4.4
                elif any(k in path for k in ("walk_speed", "boost_speed", "energy", "turn_rate")):
                    s += 3.8
                elif any(k in path for k in ("vision", "near_range", "fov")):
                    s += 2.6
            if any(k in text for k in ("armor", "damage taken", "hp", "regen", "shield")):
                s += 3.0
            if any(k in text for k in ("damage", "fire rate", "cooldown", "charge", "chain")):
                s += 2.5
            if any(k in text for k in ("speed", "energy", "turn", "boost")):
                s += 1.8
            if s > best_s:
                best_s, best_i = s, i
        return best_i

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
        rank = {"lightning": 5, "laser": 4, "mortar": 3, "gun": 1}
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
                s = d + float(m.get("hp", 700.0)) * 0.14
                if i < 5 and self._los(mp, m):
                    s -= 70.0
                if m.get("charging"):
                    s -= 40.0
                if self.last_attacker_id is not None and int(m["id"]) == self.last_attacker_id:
                    s -= 30.0
                if s < best_s:
                    best, best_s = m, s
            return best, "mech"

        ticks_quiet = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        if hp_frac < 0.50 and ticks_quiet >= 40:
            return None, "none"

        in_zone = me.get("in_zone", True)
        crates = obs.get("crates") or []
        if crates and in_zone:
            c = min(crates, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(c["pos"], mp) < 60.0 and self._los(mp, c, 1.2):
                return c, "crate"

        npcs = [m for m in mechs if m.get("kind") == "npc"]
        if npcs and in_zone and hp_frac > 0.45:
            n = min(npcs, key=lambda q: dist3d(q["pos"], mp))
            if dist2d(n["pos"], mp) < 62.0 and self._los(mp, n, 1.8):
                return n, "npc"

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
                if float(w.get("reload", 0.0)) == 0.0 and (ammo == 0 or (target is None and ammo <= 4)):
                    reload_cmd = True

        if target is None:
            if self.last_attacker_bearing is not None and tick - self.last_damage_tick < 45:
                yaw = self.last_attacker_bearing
            elif self.last_sound_bearing is not None and tick - self.last_sound_tick < 30:
                yaw = self.last_sound_bearing
            elif math.hypot(move_dir[0], move_dir[1]) > 0.1:
                yaw = math.atan2(move_dir[1], move_dir[0])
            else:
                yaw = wrap_angle(ayaw + 0.28)
            return [yaw, 0.0], fire, reload_cmd

        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target.get("vel") or [0.0, 0.0, 0.0]
        tcubes = float(target.get("cubes", 0.0))
        th = (4.5 * (1.0 + 0.05 * tcubes)) if kind == "mech" else (1.8 if kind == "npc" else 1.0)
        hd = max(1.0, dist2d((mx, my), (tx, ty)))
        has_los = self._los(me["pos"], target, th)

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

        ticks_quiet = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        regen_hiding = (hp_frac < 0.45 and not has_los and ticks_quiet >= 40)

        for si in (0, 1):
            w = weapons[si]
            if w is None:
                continue
            wk = w["kind"]
            st = w.get("stats") or {}
            cd = float(w.get("cooldown", 0.0))

            if wk == "gun" and not regen_hiding:
                rng = float(st.get("range", 230.0))
                ammo = float(w.get("ammo", 1.0))
                if (
                    ammo > 0
                    and float(w.get("reload", 0.0)) == 0.0
                    and hd <= rng
                    and angle_diff(ayaw, gun_yaw) < 0.22
                    and abs(apitch - gun_pitch) < 0.25
                    and (has_los or hd < 26.0)
                ):
                    fire[si] = True

            elif wk == "lightning" and not regen_hiding:
                rng = float(st.get("range", 45.0))
                cone = math.radians(float(st.get("cone_deg", 30.0)) * 0.5)
                cost = float(st.get("energy", 3.0))
                if (
                    cd <= 0.05
                    and me["energy"] >= cost + 3.0
                    and hd <= rng - 0.3
                    and angle_diff(ayaw, dir_yaw) <= cone
                    and has_los
                ):
                    fire[si] = True

            elif wk == "laser" and not regen_hiding:
                rng = float(st.get("range", 280.0))
                full = float(st.get("charge", 0.75))
                minc = float(st.get("min_charge", 0.15))
                ch = float(w.get("charge", 0.0))
                lx, ly, lz = tx + tvx * 0.05, ty + tvy * 0.05, tz + th + tvz * 0.05
                l_yaw = math.atan2(ly - my, lx - mx)
                l_pitch = clamp(math.atan2(lz - mz - muz, max(1.0, math.hypot(lx - mx, ly - my))), min_p, max_p)
                ey, ep = angle_diff(ayaw, l_yaw), abs(apitch - l_pitch)
                enemy_has_lightning = ("lightning" in (target.get("weapons") or [])) and hd < 44.0

                if ch > 0.0:
                    if ch >= full - 0.15 or not gun or float(gun.get("ammo", 0.0)) == 0.0:
                        override, override_prio = (l_yaw, l_pitch), 3
                    on = ey <= 0.11 and ep <= 0.15 and has_los and hd <= rng
                    thp = float(target.get("hp", 500.0))
                    if ch >= full - 0.07 and on:
                        fire[si] = False
                    elif ch >= minc + 0.20 and on and thp <= 160.0:
                        fire[si] = False
                    elif ch >= full + 1.2:
                        fire[si] = False
                    elif not has_los and ch >= minc + 0.35:
                        fire[si] = False
                    else:
                        fire[si] = True
                elif kind == "mech" and not enemy_has_lightning and cd <= 0.05 and hd <= rng and (has_los or hd < 70.0) and ey < 0.45:
                    fire[si] = True

            elif wk == "mortar" and kind in ("mech", "npc") and not regen_hiding:
                v = float(st.get("speed", 72.0))
                g = float(st.get("gravity", 22.0))
                splash = float(st.get("splash_radius", 7.0))
                gun_ready = bool(gun) and float(gun.get("ammo", 1.0)) > 0 and float(gun.get("reload", 0.0)) == 0.0
                want_mortar = (not has_los) or hd > 105.0 or not gun_ready
                if want_mortar and hd > splash + 10.0 and cd <= 0.08 and override_prio < 3:
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

    def _choose_stakeout_post(self, obs, nx: float, ny: float, nr: float) -> Tuple[float, float]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        tick = int(obs["tick"])
        if nr < 30.0:
            return nx, ny

        if (
            self.stakeout_pt is not None
            and tick < self.stakeout_until
            and dist2d(self.stakeout_pt, (nx, ny)) <= nr * 0.78
        ):
            return self.stakeout_pt

        anchor_r = clamp(nr * 0.50, 20.0, 165.0)
        ax = nx + math.cos(self.roam_angle) * anchor_r
        ay = ny + math.sin(self.roam_angle) * anchor_r
        gx, gy = int(ax // self.world.cs), int(ay // self.world.cs)

        best_pt = None
        best_sc = 1e9
        for dx_g in (-2, -1, 0, 1, 2):
            for dy_g in (-2, -1, 0, 1, 2):
                for c_idx in self.world.cover_grid.get((gx + dx_g, gy + dy_g), []):
                    wx, wy, pcx, pcy = self.world.cover_posts[c_idx]
                    dn = dist2d((wx, wy), (nx, ny))
                    if dn > nr * 0.75 or (nr > 90.0 and dn < nr * 0.25):
                        continue
                    shield_bonus = -15.0 if dn > dist2d((pcx, pcy), (nx, ny)) else 0.0
                    sc = dist2d((mx, my), (wx, wy)) * 0.6 + dist2d((ax, ay), (wx, wy)) * 0.4 + shield_bonus
                    if sc < best_sc:
                        best_sc = sc
                        best_pt = (wx, wy)

        if best_pt is None:
            best_pt = (ax, ay)
        self.stakeout_pt = best_pt
        self.stakeout_until = tick + 90
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

        if not me["in_zone"] or d_cur > max(8.0, cr - 18.0):
            sx, sy = self._choose_stakeout_post(obs, nx, ny, nr)
            return sx, sy, "zone-urgent", energy > 20.0
        if float(zone["shrink_start"]) - float(obs["t"]) < 12.0 and d_next > max(12.0, nr * 0.82):
            sx, sy = self._choose_stakeout_post(obs, nx, ny, nr)
            return sx, sy, "zone-rotate", energy > 45.0

        enemies = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
        close_enemies = [e for e in enemies if dist2d(e["pos"], (mx, my)) < 115.0]
        weapons = me.get("weapons") or [None, None]

        if len(close_enemies) <= 1:
            for d in obs.get("drops") or []:
                dx, dy, _ = d["pos"]
                if dist2d((dx, dy), (nx, ny)) > nr * 0.95 or dist2d((dx, dy), (cx, cy)) > cr - 12.0:
                    continue
                dd = dist2d((mx, my), (dx, dy))
                li = float(d.get("lands_in", 0.0))
                holder = d.get("holder")
                if dd - li * 11.5 < 95.0 and (holder is None or holder == me["id"] or hp_frac > 0.55):
                    return dx, dy, f"drop({li:.0f}s)", (li <= 2.0 and dd > 15.0 and energy > 35.0)

        cubes = [
            c for c in (obs.get("cubes") or [])
            if dist2d(c["pos"], (cx, cy)) < cr - 10.0 and abs(c["pos"][2] - mz) < 6.0
        ]
        if cubes:
            c = min(cubes, key=lambda q: dist2d(q["pos"], (mx, my)))
            cdist = dist2d(c["pos"], (mx, my))
            if cdist < (18.0 if close_enemies else 52.0):
                return c["pos"][0], c["pos"][1], "cube", False

        if weapons[1] is None:
            gp = [
                p for p in (obs.get("pickups") or [])
                if p["kind"] != "gun"
                and dist2d(p["pos"], (cx, cy)) < cr - 10.0
                and abs(p["pos"][2] - mz) < 6.0
                and tick - self.failed_pickups.get(int(p["id"]), -999) > 25
            ]
            if gp:
                gp.sort(key=lambda p: (0 if p["kind"] in ("lightning", "laser") else 1, dist2d(p["pos"], (mx, my))))
                if dist2d(gp[0]["pos"], (mx, my)) < (25.0 if close_enemies else 65.0):
                    return gp[0]["pos"][0], gp[0]["pos"][1], f"loot-{gp[0]['kind']}", False

        if target is not None and kind == "mech":
            tx, ty, tz = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            held = {w["kind"] for w in weapons if w}
            ew = target.get("weapons") or []

            gun_reloading = any(w and w["kind"] == "gun" and float(w.get("reload", 0.0)) > 0.35 for w in weapons)
            sec_ready = weapons[1] is not None and float(weapons[1].get("cooldown", 0.0)) <= 0.15 and (
                weapons[1]["kind"] != "lightning" or d <= 44.0
            )

            need_cover = (hp_frac < 0.44) or (len(close_enemies) >= 2) or (gun_reloading and not sec_ready)
            if need_cover:
                cov = self.world.find_combat_cover(mx, my, mz, tx, ty, tz, cx, cy, cr, want_los=False)
                if cov is not None:
                    return cov[0], cov[1], "cover-hide", (hp_frac < 0.38 and energy > 30.0)

            peek = self.world.find_combat_cover(mx, my, mz, tx, ty, tz, cx, cy, cr, want_los=True)
            if peek is not None and not ("lightning" in held and d < 38.0):
                jiggle = 1.0 if ((tick // 10 + self.slot) % 2 == 0) else -1.0
                rx, ry = (tx - mx) / d, (ty - my) / d
                sx, sy = -ry * jiggle, rx * jiggle
                gx = peek[0] + sx * 4.5
                gy = peek[1] + sy * 4.5
                return gx, gy, "stakeout-peek", bool(target.get("charging")) and energy > 30.0

            if "lightning" in ew and "lightning" not in held:
                pref = 64.0
            elif "lightning" in held:
                pref = 30.0
            elif "laser" in held:
                pref = 65.0
            else:
                pref = 50.0
            if need_cover:
                pref = 95.0

            sign = 1.0 if ((tick // 10 + self.slot) % 2 == 0) else -1.0
            rx, ry = (tx - mx) / d, (ty - my) / d
            sx, sy = -ry * sign, rx * sign
            radial = clamp((d - pref) / 22.0, -0.9, 0.85)
            gx = mx + (rx * radial + sx * 0.85) * 20.0
            gy = my + (ry * radial + sy * 0.85) * 20.0
            gd = dist2d((gx, gy), (cx, cy))
            if gd > cr - 14.0:
                k = max(0.0, (cr - 14.0)) / max(1.0, gd)
                gx, gy = cx + (gx - cx) * k, cy + (gy - cy) * k
            return gx, gy, "combat", bool(target.get("charging")) and energy > 30.0

        if target is not None and kind in ("crate", "npc"):
            tx, ty, _ = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            stop_d = 8.0 if kind == "crate" else 20.0
            if d > stop_d:
                return tx, ty, f"farm-{kind}", False
            return mx, my, f"farm-{kind}", False

        if tick % 300 == 0:
            self.roam_angle = wrap_angle(self.roam_angle + self.rng.uniform(0.6, 1.4))
            self.stakeout_pt = None
        sx, sy = self._choose_stakeout_post(obs, nx, ny, nr)
        return sx, sy, "stakeout", False

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
                    self.last_attacker_bearing = float(ev["bearing"])
                if ev.get("by") is not None:
                    self.last_attacker_id = int(ev["by"])
                self.last_dmg = (str(ev.get("weapon")), float(ev.get("amount", 0.0)), bool(ev.get("rear")))
            elif et == "hit":
                self.last_dealt_tick = tick
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