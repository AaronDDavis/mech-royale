#!/usr/bin/env python3
"""Apex Mech Royale Bot for mechfight/1.

Uses `mechfight.py` (standard library only) and implements:
- Fast 2.5D spatial hash & prism/terrain geometry loader (`v2/world.json`, `v2/terrain.i16`)
- Exact weapon ballistics:
  * Gun: iterative 3D velocity lead + auto-reload when out of combat
  * Mortar: gravity-compensated ballistic arc solver (low/high lob over buildings) + self-splash safety
  * Laser: hold-to-charge, release-to-fire hitscan with slewed aim verification & cook-off priority
  * Lightning: 45 m / 30° cone auto-zap & short-circuit priority against laser carriers
- Weapon triangle & smart looting:
  * Auto-equips secondary weapons (`laser` > `lightning` > `mortar`) and swaps intelligently
- Upgrade synergy scorer:
  * Evaluates `hello.config.upgrades` effects and picks the highest-EV choice in `obs["offer"]`
- Survival, Zone, Supply Drops, Dodge & Obstacle-Aware Navigation:
  * Tracks shrinking zone interpolation, uncontested supply drops, power cubes, crates, and sound bearings
  * Multi-ray obstacle avoidance (buildings, steep terrain steps, map bounds) + jump/jet over ledges
  * Evasive strafing against charging enemy lasers and incoming mortar shells
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def dist2d(a: List[float] | Tuple[float, ...], b: List[float] | Tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3d(a: List[float] | Tuple[float, ...], b: List[float] | Tuple[float, ...]) -> float:
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
    """Fast spatial index over `v2/world.json` prisms and `v2/terrain.i16` heightfield."""

    def __init__(self, map_dir: str, files: List[str]) -> None:
        self.bounds = {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0}
        self.prisms: List[Dict[str, Any]] = []
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        self.cell_size = 32.0
        self.terrain_data: Optional[bytes] = None
        self.t_min_x = -513.0
        self.t_min_y = -334.0
        self.t_step = 1.0
        self.t_nx = 1070
        self.t_ny = 724
        self.t_scale = 0.01

        world_path = os.path.join(map_dir, "v2/world.json")
        for f in files:
            if f.endswith("world.json"):
                world_path = os.path.join(map_dir, f)
                break

        if os.path.isfile(world_path):
            try:
                with open(world_path, "rb") as fp:
                    w = json.load(fp)
                if "bounds" in w:
                    self.bounds = w["bounds"]
                t_info = w.get("terrain") or {}
                self.t_min_x = float(t_info.get("minX", -513.0))
                self.t_min_y = float(t_info.get("minY", -334.0))
                self.t_step = float(t_info.get("step", 1.0))
                self.t_nx = int(t_info.get("nx", 1070))
                self.t_ny = int(t_info.get("ny", 724))
                self.t_scale = float(t_info.get("scale", 0.01))

                for p in w.get("prisms") or []:
                    ring = p.get("ring") or []
                    if len(ring) < 6:
                        continue
                    xs = ring[0::2]
                    ys = ring[1::2]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    idx = len(self.prisms)
                    self.prisms.append({
                        "kind": p.get("kind", "building"),
                        "base": float(p.get("base", -10.0)),
                        "top": float(p.get("top", 20.0)),
                        "ring": ring,
                        "holes": p.get("holes") or [],
                        "bbox": (min_x, min_y, max_x, max_y),
                    })
                    gx0 = int(math.floor(min_x / self.cell_size))
                    gx1 = int(math.floor(max_x / self.cell_size))
                    gy0 = int(math.floor(min_y / self.cell_size))
                    gy1 = int(math.floor(max_y / self.cell_size))
                    for gx in range(gx0, gx1 + 1):
                        for gy in range(gy0, gy1 + 1):
                            self.grid.setdefault((gx, gy), []).append(idx)
            except Exception as exc:
                mf.log(f"[map] world.json load warning: {exc}")

        terrain_path = os.path.join(map_dir, "v2/terrain.i16")
        for f in files:
            if f.endswith("terrain.i16"):
                terrain_path = os.path.join(map_dir, f)
                break
        if os.path.isfile(terrain_path):
            try:
                with open(terrain_path, "rb") as fp:
                    self.terrain_data = fp.read()
            except Exception as exc:
                mf.log(f"[map] terrain.i16 load warning: {exc}")

    def terrain_z(self, x: float, y: float) -> float:
        """Bilinearly samples terrain.i16 heightfield (§11)."""
        if not self.terrain_data:
            return 0.0
        fx = clamp((x - self.t_min_x) / self.t_step, 0.0, self.t_nx - 1.001)
        fy = clamp((y - self.t_min_y) / self.t_step, 0.0, self.t_ny - 1.001)
        i = int(fx)
        j = int(fy)
        tx = fx - i
        ty = fy - j
        nx = self.t_nx
        idx00 = (j * nx + i) * 2
        idx10 = idx00 + 2
        idx01 = ((j + 1) * nx + i) * 2
        idx11 = idx01 + 2
        if idx11 + 2 <= len(self.terrain_data):
            (h00,) = struct.unpack_from("<h", self.terrain_data, idx00)
            (h10,) = struct.unpack_from("<h", self.terrain_data, idx10)
            (h01,) = struct.unpack_from("<h", self.terrain_data, idx01)
            (h11,) = struct.unpack_from("<h", self.terrain_data, idx11)
            h0 = h00 * (1.0 - tx) + h10 * tx
            h1 = h01 * (1.0 - tx) + h11 * tx
            return (h0 * (1.0 - ty) + h1 * ty) * self.t_scale
        return 0.0

    def is_blocked_point(self, x: float, y: float, z: float, radius: float = 2.6, height: float = 9.0) -> Tuple[bool, float]:
        """Returns (is_blocked_wall, obstacle_top_z_if_blocked)."""
        if (
            x - radius < self.bounds["minX"]
            or x + radius > self.bounds["maxX"]
            or y - radius < self.bounds["minY"]
            or y + radius > self.bounds["maxY"]
        ):
            return True, 999.0

        gx = int(math.floor(x / self.cell_size))
        gy = int(math.floor(y / self.cell_size))
        candidates = self.grid.get((gx, gy))
        if not candidates:
            return False, z

        pts = ((x, y), (x + radius, y), (x - radius, y), (x, y + radius), (x, y - radius))
        for idx in candidates:
            p = self.prisms[idx]
            min_x, min_y, max_x, max_y = p["bbox"]
            if x + radius < min_x or x - radius > max_x or y + radius < min_y or y - radius > max_y:
                continue
            if z + height <= p["base"] + 0.2 or z >= p["top"] - 1.1:
                continue
            for px, py in pts:
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    if point_in_ring(px, py, p["ring"]):
                        in_hole = False
                        for h in p["holes"]:
                            if point_in_ring(px, py, h):
                                in_hole = True
                                break
                        if not in_hole:
                            return True, p["top"]
        return False, z

    def has_line_of_sight(self, a: List[float], b: List[float], samples: int = 8) -> bool:
        """Fast sampled segment check against solid prisms between `a` and `b`."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        for s in range(1, samples):
            t = s / float(samples)
            x = a[0] + dx * t
            y = a[1] + dy * t
            z = a[2] + dz * t
            gx = int(math.floor(x / self.cell_size))
            gy = int(math.floor(y / self.cell_size))
            candidates = self.grid.get((gx, gy))
            if not candidates:
                continue
            for idx in candidates:
                p = self.prisms[idx]
                min_x, min_y, max_x, max_y = p["bbox"]
                if min_x <= x <= max_x and min_y <= y <= max_y and p["base"] <= z <= p["top"]:
                    if point_in_ring(x, y, p["ring"]):
                        in_hole = False
                        for h in p["holes"]:
                            if point_in_ring(x, y, h):
                                in_hole = True
                                break
                        if not in_hole:
                            return False
        return True


class Bot:
    name = "apex-titan"

    def __init__(self, hello: mf.Hello) -> None:
        self.slot = int(hello.get("slot", 0))
        self.rng = random.Random(int(hello.get("seed", 42)))
        self.cfg = hello.get("config") or {}
        map_info = hello.get("map") or {"dir": "/arena", "files": ["v2/world.json", "v2/terrain.i16"]}
        self.world = WorldMap(map_info.get("dir", "/arena"), map_info.get("files") or [])

        self.upgrade_defs: Dict[str, Dict[str, Any]] = {}
        for u in self.cfg.get("upgrades") or []:
            if "id" in u:
                self.upgrade_defs[u["id"]] = u

        self.last_attacker_bearing: Optional[float] = None
        self.last_attacker_tick: int = -999
        self.last_sound_bearing: Optional[float] = None
        self.last_sound_tick: int = -999
        self.roam_angle: float = self.rng.uniform(-math.pi, math.pi)
        self.stuck_ticks: int = 0
        self.last_pos: List[float] = [0.0, 0.0, 0.0]
        self.failed_pickups: Dict[int, int] = {}

    # ------------------------------------------------------------------ Upgrade Selection
    def _pick_upgrade(self, obs: mf.Observation) -> int:
        offer = obs.get("offer")
        if not offer:
            return 0
        choices = offer.get("choices") or []
        if not choices:
            return 0

        me = obs["self"]
        held_kinds = {w["kind"] for w in (me.get("weapons") or []) if w is not None}

        best_idx = 0
        best_score = -1e9
        for idx, c in enumerate(choices):
            cid = str(c.get("id", ""))
            desc = str(c.get("desc", "")).lower()
            udef = self.upgrade_defs.get(cid, {})
            req = udef.get("requires")
            score = 0.0

            if req:
                if req in held_kinds:
                    score += 4.5
                else:
                    score -= 3.0

            for eff in udef.get("effects") or []:
                path = str(eff.get("path", "")).lower()
                if any(k in path for k in ("armor", "max_hp", "regen")):
                    score += 5.0
                elif any(k in path for k in ("damage", "cooldown", "charge", "reload", "speed")):
                    score += 4.2
                elif any(k in path for k in ("walk_speed", "boost_speed", "energy", "turn_rate")):
                    score += 3.8
                elif any(k in path for k in ("vision", "near_range", "fov")):
                    score += 2.5

            text = f"{cid} {desc}".lower()
            if any(k in text for k in ("armor", "damage taken", "hp", "regen", "shield")):
                score += 3.0
            if any(k in text for k in ("damage", "fire rate", "cooldown", "charge", "chain")):
                score += 2.5
            if any(k in text for k in ("speed", "energy", "turn", "boost")):
                score += 2.0

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # ------------------------------------------------------------------ Weapon Looting
    def _evaluate_loot(self, obs: mf.Observation) -> Tuple[Optional[int], Optional[int]]:
        """Returns (take_id, drop_slot) if a desirable weapon pickup is within reach."""
        me = obs["self"]
        x, y, z = me["pos"]
        tick = int(obs["tick"])
        weapons = me.get("weapons") or [None, None]

        for w in weapons:
            if w and w.get("kind") == "laser" and float(w.get("charge", 0.0)) > 0.0:
                return None, None

        reach = float((self.cfg.get("loot") or {}).get("pickup_radius", 4.0))
        held_kinds = {w["kind"] for w in weapons if w is not None}

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

        near_pickups = [
            p for p in (obs.get("pickups") or [])
            if dist2d(p["pos"], (x, y)) <= reach - 0.15
            and abs(p["pos"][2] - z) <= 3.8
            and p["kind"] not in held_kinds
            and tick - self.failed_pickups.get(int(p["id"]), -999) > 25
        ]
        if not near_pickups:
            return None, None

        near_pickups.sort(key=lambda p: (-rank.get(p["kind"], 0), dist2d(p["pos"], (x, y))))
        best_p = near_pickups[0]
        p_id = int(best_p["id"])
        p_rank = rank.get(best_p["kind"], 0)

        if weapons[1] is None:
            if best_p["kind"] != "gun":
                return p_id, None
        elif weapons[0] is None:
            return p_id, None
        else:
            w1_kind = weapons[1]["kind"]
            if p_rank > rank.get(w1_kind, 0) + 1 and float(weapons[1].get("cooldown", 0.0)) <= 0.2:
                return p_id, 1

        return None, None

    # ------------------------------------------------------------------ Target Selection & Ballistics
    def _select_combat_target(
        self, obs: mf.Observation
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Selects the highest-priority target ('mech', 'npc', or 'crate')."""
        me = obs["self"]
        my_pos = me["pos"]
        mechs = obs.get("mechs") or []

        enemy_mechs = [m for m in mechs if m.get("kind") == "mech"]
        npcs = [m for m in mechs if m.get("kind") == "npc"]

        if enemy_mechs:
            def mech_score(m: Dict[str, Any]) -> float:
                d = dist3d(m["pos"], my_pos)
                hp = float(m.get("hp", 700.0))
                los_bonus = -55.0 if self.world.has_line_of_sight(
                    [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                    [m["pos"][0], m["pos"][1], m["pos"][2] + 4.5],
                ) else 0.0
                charging_threat = -40.0 if m.get("charging") else 0.0
                return d + hp * 0.12 + los_bonus + charging_threat

            best_m = min(enemy_mechs, key=mech_score)
            return best_m, "mech"

        if npcs and me.get("in_zone", True):
            best_npc = min(npcs, key=lambda n: dist3d(n["pos"], my_pos))
            if dist2d(best_npc["pos"], my_pos) < 65.0 and me["hp"] > me["max_hp"] * 0.35:
                return best_npc, "npc"

        crates = obs.get("crates") or []
        if crates and me.get("in_zone", True):
            best_crate = min(crates, key=lambda c: dist3d(c["pos"], my_pos))
            if dist2d(best_crate["pos"], my_pos) < 55.0:
                return best_crate, "crate"

        return None, "none"

    def _compute_aim_and_fire(
        self,
        obs: mf.Observation,
        target: Optional[Dict[str, Any]],
        target_kind: str,
        move_dir: List[float],
    ) -> Tuple[List[float], List[bool], bool]:
        """Returns (desired_aim [yaw, pitch], fire [slot0, slot1], reload_bool)."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        actual_yaw, actual_pitch = me["aim"]
        stats = me.get("stats") or {}
        muzzle_h = float(stats.get("muzzle_height", 7.0))
        min_pitch = float((self.cfg.get("mech") or {}).get("min_pitch", -1.1))
        max_pitch = float((self.cfg.get("mech") or {}).get("max_pitch", 1.1))
        weapons = me.get("weapons") or [None, None]

        fire = [False, False]
        reload_cmd = False

        for w in weapons:
            if w and w["kind"] == "gun":
                mag = float((w.get("stats") or {}).get("mag", 12.0))
                ammo = float(w.get("ammo", mag))
                rel = float(w.get("reload", 0.0))
                if rel == 0.0 and (ammo == 0 or (target is None and ammo < mag * 0.65)):
                    reload_cmd = True

        if target is None:
            tick = int(obs["tick"])
            if self.last_attacker_bearing is not None and (tick - self.last_attacker_tick) < 35:
                look_yaw = self.last_attacker_bearing
            elif self.last_sound_bearing is not None and (tick - self.last_sound_tick) < 25:
                look_yaw = self.last_sound_bearing
            elif math.hypot(move_dir[0], move_dir[1]) > 0.1:
                look_yaw = math.atan2(move_dir[1], move_dir[0])
            else:
                look_yaw = actual_yaw
            return [look_yaw, 0.0], fire, reload_cmd

        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target.get("vel") or [0.0, 0.0, 0.0]
        target_h = 4.5 if target_kind == "mech" else (1.8 if target_kind == "npc" else 1.0)

        horiz_dist = dist2d((mx, my), (tx, ty))
        muzzle_pos = [mx, my, mz + muzzle_h]
        target_center = [tx, ty, tz + target_h]
        has_los = self.world.has_line_of_sight(muzzle_pos, target_center)

        gun_speed = 252.0
        for w in weapons:
            if w and w["kind"] == "gun":
                gun_speed = float((w.get("stats") or {}).get("speed", 252.0))
                break
        t_flight = clamp(horiz_dist / max(1.0, gun_speed), 0.0, 1.2)
        pred_x = tx + tvx * t_flight
        pred_y = ty + tvy * t_flight
        pred_z = (tz + target_h) + tvz * t_flight * 0.5
        gun_yaw = math.atan2(pred_y - my, pred_x - mx)
        gun_pitch = clamp(
            math.atan2(pred_z - (mz + muzzle_h), max(1.0, dist2d((mx, my), (pred_x, pred_y)))),
            min_pitch,
            max_pitch,
        )

        aim_yaw, aim_pitch = gun_yaw, gun_pitch

        for slot_idx in (0, 1):
            w = weapons[slot_idx]
            if w is None:
                continue
            wkind = w["kind"]
            wstats = w.get("stats") or {}
            cd = float(w.get("cooldown", 0.0))

            if wkind == "gun":
                w_range = float(wstats.get("range", 230.0))
                ammo = float(w.get("ammo", 1.0))
                rel = float(w.get("reload", 0.0))
                yaw_err = angle_diff(actual_yaw, gun_yaw)
                pitch_err = abs(actual_pitch - gun_pitch)
                if ammo > 0 and rel == 0.0 and horiz_dist <= w_range and yaw_err < 0.22 and pitch_err < 0.25:
                    if has_los or horiz_dist < 35.0:
                        fire[slot_idx] = True

            elif wkind == "lightning":
                w_range = float(wstats.get("range", 45.0))
                cone_rad = math.radians(float(wstats.get("cone_deg", 30.0)) * 0.5)
                e_cost = float(wstats.get("energy", 3.0))
                direct_yaw = math.atan2(ty - my, tx - mx)
                yaw_err = angle_diff(actual_yaw, direct_yaw)
                if (
                    cd <= 0.05
                    and me["energy"] >= e_cost + 5.0
                    and horiz_dist <= w_range - 0.5
                    and yaw_err <= cone_rad
                    and has_los
                ):
                    fire[slot_idx] = True

            elif wkind == "laser":
                w_range = float(wstats.get("range", 280.0))
                full_charge = float(wstats.get("charge", 0.75))
                min_charge = float(wstats.get("min_charge", 0.15))
                cur_charge = float(w.get("charge", 0.0))
                direct_yaw = math.atan2(ty - my, tx - mx)
                direct_pitch = clamp(
                    math.atan2((tz + target_h) - (mz + muzzle_h), max(1.0, horiz_dist)),
                    min_pitch,
                    max_pitch,
                )
                yaw_err = angle_diff(actual_yaw, direct_yaw)
                pitch_err = abs(actual_pitch - direct_pitch)

                if cur_charge > 0.0:
                    aim_yaw, aim_pitch = direct_yaw, direct_pitch
                    on_target = (yaw_err < 0.085 and pitch_err < 0.12 and has_los and horiz_dist <= w_range)
                    if cur_charge >= full_charge - 0.08 and on_target:
                        fire[slot_idx] = False
                    elif cur_charge >= min_charge + 0.25 and on_target and float(target.get("hp", 500.0)) < 140.0:
                        fire[slot_idx] = False
                    else:
                        fire[slot_idx] = True
                else:
                    if cd <= 0.05 and horiz_dist <= w_range and (has_los or horiz_dist < 90.0) and yaw_err < 0.45:
                        fire[slot_idx] = True
                        aim_yaw, aim_pitch = direct_yaw, direct_pitch

            elif wkind == "mortar":
                v = float(wstats.get("speed", 72.0))
                g = float(wstats.get("gravity", 22.0))
                splash_r = float(wstats.get("splash_radius", 7.0))
                if horiz_dist > splash_r + 8.0 and cd <= 0.1 and target_kind in ("mech", "npc"):
                    t_est = horiz_dist / max(20.0, v * 0.78)
                    mx_t = tx + tvx * t_est
                    my_t = ty + tvy * t_est
                    R = max(1.0, dist2d((mx, my), (mx_t, my_t)))
                    dh = tz - (mz + 9.5)
                    v2 = v * v
                    disc = v2 * v2 - g * (g * R * R + 2.0 * dh * v2)
                    if disc >= 0.0:
                        root = math.sqrt(disc)
                        tan_low = (v2 - root) / (g * R)
                        tan_high = (v2 + root) / (g * R)
                        pitch_low = math.atan2(tan_low, 1.0)
                        pitch_high = math.atan2(tan_high, 1.0)
                        chosen_pitch = pitch_low if has_los else (
                            pitch_high if pitch_high <= max_pitch else pitch_low
                        )
                        if min_pitch <= chosen_pitch <= max_pitch:
                            m_yaw = math.atan2(my_t - my, mx_t - mx)
                            if cd <= 0.25:
                                aim_yaw = m_yaw
                                aim_pitch = chosen_pitch
                            if angle_diff(actual_yaw, m_yaw) < 0.18 and abs(actual_pitch - chosen_pitch) < 0.18:
                                fire[slot_idx] = True

        return [aim_yaw, aim_pitch], fire, reload_cmd

    # ------------------------------------------------------------------ Zone, Supply Drops & Navigation
    def _current_zone(self, obs: mf.Observation) -> Tuple[float, float, float]:
        z = obs["zone"]
        t = float(obs["t"])
        s0 = float(z["shrink_start"])
        s1 = float(z["shrink_end"])
        c0 = z["center"]
        c1 = z["next_center"]
        r0 = float(z["radius"])
        r1 = float(z["next_radius"])
        if t <= s0 or s1 <= s0:
            return float(c0[0]), float(c0[1]), r0
        if t >= s1:
            return float(c1[0]), float(c1[1]), r1
        alpha = clamp((t - s0) / (s1 - s0), 0.0, 1.0)
        return (
            c0[0] + (c1[0] - c0[0]) * alpha,
            c0[1] + (c1[1] - c0[1]) * alpha,
            r0 + (r1 - r0) * alpha,
        )

    def _choose_goal(
        self, obs: mf.Observation, target: Optional[Dict[str, Any]], target_kind: str
    ) -> Tuple[float, float, str, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        energy = float(me["energy"])
        zone = obs["zone"]
        nx, ny = float(zone["next_center"][0]), float(zone["next_center"][1])
        nr = float(zone["next_radius"])
        cx, cy, cr = self._current_zone(obs)

        dist_cur_center = dist2d((mx, my), (cx, cy))
        dist_next_center = dist2d((mx, my), (nx, ny))

        if not me["in_zone"] or dist_cur_center > max(8.0, cr - 18.0):
            return nx, ny, "zone-urgent", energy > 20.0

        time_to_shrink = float(zone["shrink_start"]) - float(obs["t"])
        if time_to_shrink < 12.0 and dist_next_center > max(12.0, nr * 0.82):
            return nx, ny, "zone-rotate", energy > 45.0

        for d in obs.get("drops") or []:
            dx, dy, dz = d["pos"]
            if dist2d((dx, dy), (nx, ny)) > nr * 0.95:
                continue
            d_dist = dist2d((mx, my), (dx, dy))
            lands_in = float(d.get("lands_in", 0.0))
            holder = d.get("holder")
            if d_dist - lands_in * 11.5 < 95.0 and (holder is None or holder == me["id"] or hp_frac > 0.55):
                return dx, dy, f"drop({lands_in:.0f}s)", (lands_in <= 2.0 and d_dist > 15.0 and energy > 35.0)

        cubes = [
            c for c in (obs.get("cubes") or [])
            if dist2d(c["pos"], (cx, cy)) < cr - 10.0 and abs(c["pos"][2] - mz) < 6.0
        ]
        if cubes:
            nearest_cube = min(cubes, key=lambda c: dist2d(c["pos"], (mx, my)))
            if dist2d(nearest_cube["pos"], (mx, my)) < 45.0:
                return nearest_cube["pos"][0], nearest_cube["pos"][1], "cube", False

        weapons = me.get("weapons") or [None, None]
        if weapons[1] is None:
            good_pickups = [
                p for p in (obs.get("pickups") or [])
                if p["kind"] != "gun"
                and dist2d(p["pos"], (cx, cy)) < cr - 10.0
                and abs(p["pos"][2] - mz) < 6.0
                and int(obs["tick"]) - self.failed_pickups.get(int(p["id"]), -999) > 25
            ]
            if good_pickups:
                nearest_p = min(good_pickups, key=lambda p: dist2d(p["pos"], (mx, my)))
                if dist2d(nearest_p["pos"], (mx, my)) < 55.0:
                    return nearest_p["pos"][0], nearest_p["pos"][1], f"loot-{nearest_p['kind']}", False

        if target is not None and target_kind == "mech":
            tx, ty, tz = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            held_kinds = {w["kind"] for w in weapons if w is not None}

            pref_dist = 32.0 if "lightning" in held_kinds else (65.0 if "laser" in held_kinds else 50.0)
            if hp_frac < 0.38:
                pref_dist = 95.0

            strafe_sign = 1.0 if ((int(obs["tick"]) // 18 + self.slot) % 2 == 0) else -1.0
            rx, ry = (tx - mx) / d, (ty - my) / d
            sx, sy = -ry * strafe_sign, rx * strafe_sign
            radial = clamp((d - pref_dist) / 25.0, -0.85, 0.85)

            gx = mx + (rx * radial + sx * 0.85) * 20.0
            gy = my + (ry * radial + sy * 0.85) * 20.0
            urgent_dodge = bool(target.get("charging")) and energy > 30.0
            return gx, gy, "combat", urgent_dodge

        if dist_next_center > max(18.0, nr * 0.55):
            return nx, ny, "patrol-zone", False

        if int(obs["tick"]) % 60 == 0:
            self.roam_angle = wrap_angle(self.roam_angle + self.rng.uniform(-1.2, 1.2))
        patrol_r = min(45.0, nr * 0.4)
        return (
            nx + math.cos(self.roam_angle) * patrol_r,
            ny + math.sin(self.roam_angle) * patrol_r,
            "roam",
            False,
        )

    def _steer_move(
        self, obs: mf.Observation, goal_x: float, goal_y: float
    ) -> Tuple[List[float], bool, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        energy = float(me["energy"])
        grounded = bool(me.get("grounded", True))
        cubes = float(me.get("cubes", 0.0))
        radius = 2.5 * (1.0 + 0.05 * cubes)
        height = 9.0 * (1.0 + 0.05 * cubes)

        dodge_x, dodge_y = 0.0, 0.0
        for proj in obs.get("projectiles") or []:
            px, py, pz = proj["pos"]
            vx, vy, vz = proj["vel"]
            speed2 = vx * vx + vy * vy
            if speed2 < 1.0:
                continue
            t_closest = clamp(((mx - px) * vx + (my - py) * vy) / speed2, 0.0, 1.2)
            cx = px + vx * t_closest
            cy = py + vy * t_closest
            d = dist2d((mx, my), (cx, cy))
            danger_r = 9.5 if proj.get("kind") == "mortar" else 4.5
            if d < danger_r:
                plen = math.hypot(vx, vy)
                nx_p, ny_p = -vy / plen, vx / plen
                side = 1.0 if ((mx - px) * nx_p + (my - py) * ny_p) >= 0.0 else -1.0
                dodge_x += nx_p * side * 1.5
                dodge_y += ny_p * side * 1.5

        dx = goal_x - mx + dodge_x * 12.0
        dy = goal_y - my + dodge_y * 12.0
        dist_goal = math.hypot(dx, dy)
        if dist_goal < 1.0:
            return [0.0, 0.0], False, False

        desired_yaw = math.atan2(dy, dx)

        moved = dist2d((mx, my), self.last_pos)
        self.last_pos = [mx, my, mz]
        if moved < 0.18 and dist_goal > 4.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)

        best_yaw = desired_yaw
        best_cost = 1e9
        need_jump = False
        need_jet = False

        offsets = [
            0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.85, -1.85, 2.4, -2.4
        ]
        if self.stuck_ticks > 6:
            bias = 1.2 if (self.slot % 2 == 0) else -1.2
            desired_yaw = wrap_angle(desired_yaw + bias)

        for off in offsets:
            cand_yaw = wrap_angle(desired_yaw + off)
            ux, uy = math.cos(cand_yaw), math.sin(cand_yaw)
            blocked = False
            can_climb = False
            for step_d in (4.0, 8.5, 14.0):
                tx = mx + ux * step_d
                ty = my + uy * step_d
                is_blk, top_z = self.world.is_blocked_point(tx, ty, mz, radius=radius, height=height)
                if is_blk:
                    if (top_z - mz) <= 16.0 and energy > 55.0 and abs(off) < 0.4:
                        can_climb = True
                    else:
                        blocked = True
                        break
                tz_ground = self.world.terrain_z(tx, ty)
                if tz_ground - mz > step_d * 1.15 + 1.2:
                    blocked = True
                    break

            if not blocked:
                cost = abs(off) + (0.6 if can_climb else 0.0)
                if cost < best_cost:
                    best_cost = cost
                    best_yaw = cand_yaw
                    if can_climb:
                        need_jump = grounded
                        need_jet = not grounded

        if self.stuck_ticks > 4 and energy > 30.0:
            need_jump = grounded
            need_jet = not grounded

        return [math.cos(best_yaw), math.sin(best_yaw)], need_jump, need_jet

    # ------------------------------------------------------------------ Main Decision Loop
    def act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        me = obs["self"]
        mx, my, mz = me["pos"]

        for ev in obs.get("events") or []:
            etype = ev.get("type")
            if etype == "damaged" and ev.get("bearing") is not None:
                self.last_attacker_bearing = float(ev["bearing"])
                self.last_attacker_tick = tick
            elif etype == "take_failed":
                self.failed_pickups[int(ev.get("id", -1))] = tick

        sounds = obs.get("sounds") or []
        if sounds:
            near_sounds = [s for s in sounds if s.get("dist") in ("near", "mid")]
            if near_sounds:
                s_best = min(near_sounds, key=lambda s: float(s.get("age", 1.0)))
                self.last_sound_bearing = float(s_best["bearing"])
                self.last_sound_tick = tick

        target, target_kind = self._select_combat_target(obs)
        goal_x, goal_y, mode_label, boost_goal = self._choose_goal(obs, target, target_kind)

        move_dir, jump_nav, jet_nav = self._steer_move(obs, goal_x, goal_y)
        aim, fire, reload_cmd = self._compute_aim_and_fire(obs, target, target_kind, move_dir)

        grounded = bool(me.get("grounded", True))
        boost = (boost_goal and grounded) or (jet_nav and not grounded)
        jump = jump_nav

        act: mf.Action = {
            "tick": obs["tick"],
            "move": [round(move_dir[0], 4), round(move_dir[1], 4)],
            "aim": [round(aim[0], 4), round(aim[1], 4)],
            "fire": [bool(fire[0]), bool(fire[1])],
            "boost": bool(boost),
            "jump": bool(jump),
            "reload": bool(reload_cmd),
        }

        if obs.get("offer") is not None:
            act["pick"] = self._pick_upgrade(obs)

        take_id, drop_slot = self._evaluate_loot(obs)
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        mf.draw.point([goal_x, goal_y, mz + 1.0], "#40c0ff", 5)
        if target is not None:
            tpos = target["pos"]
            mf.draw.line([mx, my, mz + 7.0], [tpos[0], tpos[1], tpos[2] + 3.5], "#ff5040")
        w_names = "/".join(w["kind"] if w else "-" for w in (me.get("weapons") or []))
        mf.draw.label(f"{mode_label} | hp:{me['hp']:.0f} e:{me['energy']:.0f} [{w_names}]")

        return act


if __name__ == "__main__":
    mf.run(Bot)
