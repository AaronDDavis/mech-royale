#!/usr/bin/env python3
"""Apex Survivor v2 Bot for mechfight/1.

Engineered specifically to maximize placement score (`16 - place`, max 15.0):
1. SURVIVAL & ANTI-CENTER-BRAWL POSITIONING:
   - Never walks blindly into `next_center` where 10+ bots slaughter each other early.
   - Holds the quietest safe sector inside the shrinking zone (`0.52 * next_radius`), away from enemy clusters, gunfire sounds, and beam paths.
   - Stealth walking (`length <= 0.40`, `< 5 m/s` = ZERO footstep sounds!) when Rotating/Hiding near enemies out of LOS.
2. ROOFTOP HIGH-GROUND & COVER SEEKING:
   - Actively climbs medium-height rooftops (`6.5m .. 32m`) inside the safe zone when energy >= 72, granting massive cover, free HP regeneration, and downward firing angles.
   - Uses building perimeter cover when HP < 58% or when multiple enemies are visible, breaking LOS to trigger the 5-second `45 HP/s` health regeneration.
3. REAR-ARMOR SHIELDING (1.5x Damage Prevention):
   - Always faces the highest-threat visible enemy, recent damage `bearing`, or `near` weapon sound (`gun`, `laser_charge`, `lightning`, `mortar`).
   - Never turns its back to a known enemy while retreating (kites backward while facing the threat!).
4. FIXED 4-WEAPON BALLISTICS & DECOUPLED AIM:
   - Fixes the mortar/gun pitch conflict: Mortar only overrides torso pitch on the exact tick its cooldown is ready (`cooldown <= 0.12`), so the Gun (`0.24s` cooldown) and Lightning (`0.25s` cooldown) fire continuously with true 3D iterative velocity lead.
   - Laser hold-and-release controller: pre-charges on targets, tracks with hitscan precision, and releases (`fire=False`) the instant charge is sufficient and `self.aim` is aligned.
   - Respects HP regen timer: when regenerating HP at low health (`< 52%`) and undetected, holds fire on distant/non-threatening targets so dealing damage doesn't reset our 5s regen clock!
5. SMART LOOTING, CRATE/NPC FARMING & UPGRADE SYNERGY:
   - Prioritizes `armor` (-8% dmg taken), `max_hp`, `regen`, and active weapon upgrades.
   - Farms nearby crates and isolated NPCs for power cubes (+4% HP / +4% dmg per cube) only when uncontested.
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
    """Fast 2.5D spatial index over `v2/world.json` prisms and `v2/terrain.i16` heightfield."""

    def __init__(self, map_dir: str, files: List[str]) -> None:
        self.bounds = {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0}
        self.prisms: List[Dict[str, Any]] = []
        self.rooftops: List[Dict[str, Any]] = []
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
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    base = float(p.get("base", -10.0))
                    top = float(p.get("top", 20.0))
                    kind = p.get("kind", "building")
                    idx = len(self.prisms)
                    p_obj = {
                        "id": idx,
                        "kind": kind,
                        "base": base,
                        "top": top,
                        "ring": ring,
                        "holes": p.get("holes") or [],
                        "bbox": (min_x, min_y, max_x, max_y),
                        "center": (cx, cy),
                    }
                    self.prisms.append(p_obj)
                    # Index climbable rooftops with enough area to stand on
                    if kind == "building" and 6.5 <= (top - base) <= 32.0 and (max_x - min_x) >= 12.0 and (max_y - min_y) >= 12.0:
                        if point_in_ring(cx, cy, ring) and not any(point_in_ring(cx, cy, h) for h in p_obj["holes"]):
                            self.rooftops.append(p_obj)

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
            x - radius < self.bounds["minX"] + 2.0
            or x + radius > self.bounds["maxX"] - 2.0
            or y - radius < self.bounds["minY"] + 2.0
            or y + radius > self.bounds["maxY"] - 2.0
        ):
            return True, 999.0

        gx = int(math.floor(x / self.cell_size))
        gy = int(math.floor(y / self.cell_size))
        candidates = self.grid.get((gx, gy))
        if not candidates:
            return False, z

        pts = ((x, y), (x + radius, y), (x - radius, y), (x, y + radius), (x, y - radius))
        max_top = z
        blocked = False
        for idx in candidates:
            p = self.prisms[idx]
            min_x, min_y, max_x, max_y = p["bbox"]
            if x + radius < min_x or x - radius > max_x or y + radius < min_y or y - radius > max_y:
                continue
            # Bridges above head or roofs already below feet (+step_up) do not block horizontal movement
            if z + height <= p["base"] + 0.15 or z >= p["top"] - 1.15:
                continue
            for px, py in pts:
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    if point_in_ring(px, py, p["ring"]):
                        if not any(point_in_ring(px, py, h) for h in p["holes"]):
                            blocked = True
                            if p["top"] > max_top:
                                max_top = p["top"]
                            break
        return blocked, max_top

    def has_line_of_sight(self, a: List[float], b: List[float], samples: int = 10) -> bool:
        """Fast sampled 3D segment check against solid prisms between `a` and `b`."""
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
                        if not any(point_in_ring(x, y, h) for h in p["holes"]):
                            return False
        return True

    def find_cover_point(
        self, mx: float, my: float, mz: float, threat_x: float, threat_y: float, tz: float
    ) -> Optional[Tuple[float, float]]:
        """Finds a nearby walkable point behind a building prism that blocks LOS from (threat_x, threat_y)."""
        gx = int(math.floor(mx / self.cell_size))
        gy = int(math.floor(my / self.cell_size))
        best_pt: Optional[Tuple[float, float]] = None
        best_score = 1e9

        seen = set()
        for dx_g in (-1, 0, 1):
            for dy_g in (-1, 0, 1):
                for idx in self.grid.get((gx + dx_g, gy + dy_g), []):
                    if idx in seen:
                        continue
                    seen.add(idx)
                    p = self.prisms[idx]
                    if p["kind"] == "bridge" or p["top"] < mz + 5.0:
                        continue
                    pcx, pcy = p["center"]
                    d_me = dist2d((mx, my), (pcx, pcy))
                    if d_me > 65.0:
                        continue
                    # Push point on the far side of prism center away from threat
                    vx = pcx - threat_x
                    vy = pcy - threat_y
                    vlen = max(1.0, math.hypot(vx, vy))
                    ux, uy = vx / vlen, vy / vlen
                    min_x, min_y, max_x, max_y = p["bbox"]
                    half_span = 0.5 * math.hypot(max_x - min_x, max_y - min_y) + 5.0
                    cand_x = pcx + ux * half_span
                    cand_y = pcy + uy * half_span
                    blk, _ = self.is_blocked_point(cand_x, cand_y, mz, radius=3.0)
                    if not blk:
                        if not self.has_line_of_sight(
                            [cand_x, cand_y, mz + 5.0], [threat_x, threat_y, tz + 7.0], samples=8
                        ):
                            sc = dist2d((mx, my), (cand_x, cand_y))
                            if sc < best_score:
                                best_score = sc
                                best_pt = (cand_x, cand_y)
        return best_pt


class Bot:
    name = "apex-survivor"

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

        # Tactical memory across ticks
        self.last_damage_tick: int = -999
        self.last_dealt_tick: int = -999
        self.last_attacker_bearing: Optional[float] = None
        self.last_attacker_id: Optional[int] = None
        self.last_sound_bearing: Optional[float] = None
        self.last_sound_tick: int = -999
        self.danger_bearings: List[Tuple[float, int]] = []  # (yaw, tick)

        # Each bot slot gets a unique preferred angle on the safe ring to avoid spawn/center clumping
        self.safe_angle: float = wrap_angle((self.slot / 16.0) * 2.0 * math.pi + self.rng.uniform(-0.25, 0.25))
        self.stuck_ticks: int = 0
        self.last_pos: List[float] = [0.0, 0.0, 0.0]
        self.failed_pickups: Dict[int, int] = {}
        self.climb_mode_until: int = -1

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
                    score += 5.0
                else:
                    score -= 4.0

            for eff in udef.get("effects") or []:
                path = str(eff.get("path", "")).lower()
                if "armor" in path:
                    score += 8.0
                elif any(k in path for k in ("max_hp", "regen")):
                    score += 6.5
                elif any(k in path for k in ("walk_speed", "boost_speed", "energy", "turn_rate")):
                    score += 5.2
                elif any(k in path for k in ("damage", "cooldown", "charge", "reload", "speed", "range")):
                    score += 4.8
                elif any(k in path for k in ("vision", "near_range", "fov")):
                    score += 3.0

            text = f"{cid} {desc}".lower()
            if any(k in text for k in ("armor", "damage taken", "hp", "regen", "shield")):
                score += 3.5
            if any(k in text for k in ("damage", "fire rate", "cooldown", "charge", "chain")):
                score += 2.5
            if any(k in text for k in ("speed", "energy", "turn", "boost")):
                score += 2.2

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

        # Never swap/drop while a laser is charging
        for w in weapons:
            if w and w.get("kind") == "laser" and float(w.get("charge", 0.0)) > 0.0:
                return None, None

        reach = float((self.cfg.get("loot") or {}).get("pickup_radius", 4.0))
        held_kinds = {w["kind"] for w in weapons if w is not None}

        # Base weapon ranking: laser & lightning are top tier for consistent hitscan/cone damage; mortar is great secondary
        rank = {"laser": 5, "lightning": 4, "mortar": 3, "gun": 2}

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
            if dist2d(p["pos"], (x, y)) <= reach - 0.12
            and abs(p["pos"][2] - z) <= 3.8
            and p["kind"] not in held_kinds
            and tick - self.failed_pickups.get(int(p["id"]), -999) > 20
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
            # Only swap out slot 1 if we aren't mid-fight or the new weapon is strictly higher ranked
            w1_kind = weapons[1]["kind"]
            if p_rank > rank.get(w1_kind, 0) + 1 and float(weapons[1].get("cooldown", 0.0)) <= 0.15:
                return p_id, 1

        return None, None

    # ------------------------------------------------------------------ Target Selection & Ballistics
    def _select_combat_target(
        self, obs: mf.Observation
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Selects the highest-priority target ('mech', 'npc', or 'crate')."""
        me = obs["self"]
        my_pos = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        tick = int(obs["tick"])
        mechs = obs.get("mechs") or []

        enemy_mechs = [m for m in mechs if m.get("kind") == "mech"]
        npcs = [m for m in mechs if m.get("kind") == "npc"]

        # Check if we are currently regenerating HP or waiting for the 5s regen delay
        ticks_since_combat = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        trying_to_regen = (hp_frac < 0.65 and ticks_since_combat >= 40 and me.get("in_zone", True))

        if enemy_mechs:
            def mech_score(m: Dict[str, Any]) -> float:
                d = dist3d(m["pos"], my_pos)
                hp = float(m.get("hp", 700.0))
                has_los = self.world.has_line_of_sight(
                    [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                    [m["pos"][0], m["pos"][1], m["pos"][2] + 4.5],
                )
                los_bonus = -80.0 if has_los else 0.0
                # Prioritize low-HP kills and enemies aiming at us or charging lasers
                aim_at_me = angle_diff(float(m["aim"][0]), math.atan2(my_pos[1] - m["pos"][1], my_pos[0] - m["pos"][0])) < 0.45
                threat_bonus = -45.0 if (m.get("charging") or aim_at_me) else 0.0
                attacker_bonus = -35.0 if (self.last_attacker_id is not None and int(m["id"]) == self.last_attacker_id) else 0.0
                return d + hp * 0.10 + los_bonus + threat_bonus + attacker_bonus

            best_m = min(enemy_mechs, key=mech_score)
            return best_m, "mech"

        # Do not shoot NPCs or Crates if we are trying to regenerate HP (dealing damage resets the 5s regen delay!)
        if trying_to_regen:
            return None, "none"

        # Shoot crates for free cubes & weapons when safe and no enemy mechs are near
        crates = obs.get("crates") or []
        if crates and me.get("in_zone", True):
            visible_crates = [
                c for c in crates
                if dist2d(c["pos"], my_pos) < 60.0
                and self.world.has_line_of_sight(
                    [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                    [c["pos"][0], c["pos"][1], c["pos"][2] + 1.2],
                )
            ]
            if visible_crates:
                return min(visible_crates, key=lambda c: dist3d(c["pos"], my_pos)), "crate"

        # Farm isolated NPCs only if healthy and close
        if npcs and me.get("in_zone", True) and hp_frac > 0.72:
            best_npc = min(npcs, key=lambda n: dist3d(n["pos"], my_pos))
            if dist2d(best_npc["pos"], my_pos) < 48.0:
                return best_npc, "npc"

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
        mvx, mvy, mvz = me.get("vel") or [0.0, 0.0, 0.0]
        actual_yaw, actual_pitch = me["aim"]
        stats = me.get("stats") or {}
        muzzle_h = float(stats.get("muzzle_height", 7.0))
        min_pitch = float((self.cfg.get("mech") or {}).get("min_pitch", -1.1))
        max_pitch = float((self.cfg.get("mech") or {}).get("max_pitch", 1.1))
        weapons = me.get("weapons") or [None, None]
        tick = int(obs["tick"])
        hp_frac = me["hp"] / max(1.0, me["max_hp"])

        fire = [False, False]
        reload_cmd = False

        # Auto-reload gun when out of combat or low ammo with no immediate LOS target
        for w in weapons:
            if w and w["kind"] == "gun":
                mag = float((w.get("stats") or {}).get("mag", 12.0))
                ammo = float(w.get("ammo", mag))
                rel = float(w.get("reload", 0.0))
                if rel == 0.0 and (ammo == 0 or (target is None and ammo < mag * 0.75)):
                    reload_cmd = True

        if target is None:
            # Face recent damage or nearby threat sound to protect front 180° arc (prevents 1.5x rear damage!)
            if self.last_attacker_bearing is not None and (tick - self.last_damage_tick) < 45:
                look_yaw = self.last_attacker_bearing
            elif self.last_sound_bearing is not None and (tick - self.last_sound_tick) < 35:
                look_yaw = self.last_sound_bearing
            elif math.hypot(move_dir[0], move_dir[1]) > 0.08:
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

        # If we are low HP (< 52%), regenerating HP, and the enemy does NOT have LOS or isn't looking at us,
        # keep our front armor pointed at them but DO NOT fire (firing resets our 45 HP/s regeneration!)
        ticks_since_combat = min(tick - self.last_damage_tick, tick - self.last_dealt_tick)
        enemy_looking = False
        if target_kind == "mech":
            enemy_yaw = float((target.get("aim") or [0.0, 0.0])[0])
            enemy_looking = angle_diff(enemy_yaw, math.atan2(my - ty, mx - tx)) < 0.65
        hold_fire_for_regen = (
            hp_frac < 0.52
            and ticks_since_combat >= 45
            and me.get("in_zone", True)
            and (not has_los or (not enemy_looking and horiz_dist > 55.0))
            and float(target.get("hp", 500.0)) > 180.0
        )

        # Iterative 3D velocity lead for the Gun (accounts for shot origin 3.0m out along aim)
        gun_speed = 252.0
        for w in weapons:
            if w and w["kind"] == "gun":
                gun_speed = float((w.get("stats") or {}).get("speed", 252.0))
                break

        t_flight = max(0.0, (horiz_dist - 3.0) / max(1.0, gun_speed))
        for _ in range(2):
            px = tx + tvx * t_flight
            py = ty + tvy * t_flight
            pz = (tz + target_h) + tvz * t_flight * 0.5
            d3 = max(0.0, dist3d(muzzle_pos, (px, py, pz)) - 3.0)
            t_flight = clamp(d3 / max(1.0, gun_speed), 0.0, 1.1)

        pred_x = tx + tvx * t_flight
        pred_y = ty + tvy * t_flight
        pred_z = (tz + target_h) + tvz * t_flight * 0.5
        gun_yaw = math.atan2(pred_y - my, pred_x - mx)
        gun_pitch = clamp(
            math.atan2(pred_z - (mz + muzzle_h), max(1.0, dist2d((mx, my), (pred_x, pred_y)))),
            min_pitch,
            max_pitch,
        )

        direct_yaw = math.atan2(ty - my, tx - mx)
        direct_pitch = clamp(
            math.atan2((tz + target_h) - (mz + muzzle_h), max(1.0, horiz_dist)),
            min_pitch,
            max_pitch,
        )

        # Default torso aim is gun lead when in LOS, or direct yaw to keep front armor facing enemy
        aim_yaw, aim_pitch = (gun_yaw, gun_pitch) if has_los else (direct_yaw, direct_pitch)

        if hold_fire_for_regen:
            return [direct_yaw, direct_pitch], [False, False], reload_cmd

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
                # Tighter angular tolerance at long range so we don't waste mag into empty air
                max_err = clamp(0.10 + 4.5 / max(10.0, horiz_dist), 0.12, 0.28)
                if ammo > 0 and rel == 0.0 and horiz_dist <= w_range and yaw_err < max_err and pitch_err < 0.24:
                    if has_los:
                        fire[slot_idx] = True

            elif wkind == "lightning":
                w_range = float(wstats.get("range", 45.0))
                cone_rad = math.radians(float(wstats.get("cone_deg", 30.0)) * 0.5)
                e_cost = float(wstats.get("energy", 3.0))
                yaw_err = angle_diff(actual_yaw, direct_yaw)
                if (
                    cd <= 0.05
                    and me["energy"] >= e_cost + 4.0
                    and horiz_dist <= w_range - 0.3
                    and yaw_err <= cone_rad - 0.02
                    and has_los
                ):
                    fire[slot_idx] = True

            elif wkind == "laser":
                w_range = float(wstats.get("range", 280.0))
                full_charge = float(wstats.get("charge", 0.75))
                min_charge = float(wstats.get("min_charge", 0.15))
                cur_charge = float(w.get("charge", 0.0))
                # Lead slightly by 1 tick (0.05s) since release fires on the decision tick
                lx = tx + tvx * 0.05
                ly = ty + tvy * 0.05
                lz = (tz + target_h) + tvz * 0.05
                l_yaw = math.atan2(ly - my, lx - mx)
                l_pitch = clamp(
                    math.atan2(lz - (mz + muzzle_h), max(1.0, dist2d((mx, my), (lx, ly)))),
                    min_pitch,
                    max_pitch,
                )
                yaw_err = angle_diff(actual_yaw, l_yaw)
                pitch_err = abs(actual_pitch - l_pitch)

                if cur_charge > 0.0:
                    aim_yaw, aim_pitch = l_yaw, l_pitch
                    hit_tol = clamp(3.2 / max(8.0, horiz_dist), 0.035, 0.14)
                    on_target = (yaw_err <= hit_tol and pitch_err <= hit_tol * 1.3 and has_los and horiz_dist <= w_range)
                    # Release trigger (fire=False) to fire hitscan beam when fully charged or lethal
                    if cur_charge >= full_charge - 0.06 and on_target:
                        fire[slot_idx] = False
                    elif cur_charge >= min_charge + 0.20 and on_target and float(target.get("hp", 500.0)) <= 165.0:
                        fire[slot_idx] = False
                    elif cur_charge >= full_charge + 1.75:
                        # About to auto-fire at hold_s (2.0s) anyway
                        fire[slot_idx] = False
                    else:
                        fire[slot_idx] = True
                else:
                    if cd <= 0.05 and horiz_dist <= w_range and (has_los or horiz_dist < 75.0) and yaw_err < 0.42:
                        fire[slot_idx] = True
                        aim_yaw, aim_pitch = l_yaw, l_pitch

            elif wkind == "mortar":
                # Crucial fix: only point torso up for mortar when mortar is actually ready to fire (`cd <= 0.12`),
                # OR when target is behind cover (`not has_los`) so we lob over buildings without disrupting gun!
                v = float(wstats.get("speed", 72.0))
                g = float(wstats.get("gravity", 22.0))
                splash_r = float(wstats.get("splash_radius", 7.0))
                if horiz_dist > splash_r + 10.0 and cd <= 0.12 and target_kind in ("mech", "npc"):
                    t_est = horiz_dist / max(20.0, v * 0.76)
                    mx_t = tx + tvx * t_est
                    my_t = ty + tvy * t_est
                    R = max(1.0, dist2d((mx, my), (mx_t, my_t)))
                    dh = tz - (mz + 9.5)
                    v2 = v * v
                    disc = v2 * v2 - g * (g * R * R + 2.0 * dh * v2)
                    if disc >= 0.0:
                        root = math.sqrt(disc)
                        pitch_low = math.atan2((v2 - root) / (g * R), 1.0)
                        pitch_high = math.atan2((v2 + root) / (g * R), 1.0)
                        chosen_pitch = pitch_low if has_los else (
                            pitch_high if pitch_high <= max_pitch else pitch_low
                        )
                        if min_pitch <= chosen_pitch <= max_pitch:
                            m_yaw = math.atan2(my_t - my, mx_t - mx)
                            aim_yaw = m_yaw
                            aim_pitch = chosen_pitch
                            if angle_diff(actual_yaw, m_yaw) < 0.16 and abs(actual_pitch - chosen_pitch) < 0.16:
                                fire[slot_idx] = True

        return [aim_yaw, aim_pitch], fire, reload_cmd

    # ------------------------------------------------------------------ Zone, High-Ground, Cover & Navigation
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

    def _quiet_safe_point(self, obs: mf.Observation, nx: float, ny: float, nr: float) -> Tuple[float, float]:
        """Finds a safe point inside `next_center` that avoids the central moshpit and enemy clusters."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        if nr < 28.0:
            return nx, ny

        # Sample 8 candidate points on a ring at 0.52 * next_radius
        ring_r = clamp(nr * 0.52, 18.0, 175.0)
        best_pt = (nx + math.cos(self.safe_angle) * ring_r, ny + math.sin(self.safe_angle) * ring_r)
        best_score = -1e9

        enemies = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]
        for k in range(8):
            ang = wrap_angle(self.safe_angle + k * (math.pi / 4.0))
            px = nx + math.cos(ang) * ring_r
            py = ny + math.sin(ang) * ring_r
            blk, top_z = self.world.is_blocked_point(px, py, mz, radius=3.0)
            if blk and (top_z - mz) > 18.0:
                continue
            # Prefer staying near our current angle unless enemies/sounds are in that direction
            score = -dist2d((mx, my), (px, py)) * 0.25
            for e in enemies:
                score += min(140.0, dist2d((px, py), e["pos"])) * 0.9
            for b_yaw, b_tick in self.danger_bearings:
                cand_bearing = math.atan2(py - my, px - mx)
                score += angle_diff(cand_bearing, b_yaw) * 18.0
            if score > best_score:
                best_score = score
                best_pt = (px, py)
                if k != 0 and score > best_score + 25.0:
                    self.safe_angle = ang

        return best_pt

    def _choose_goal(
        self, obs: mf.Observation, target: Optional[Dict[str, Any]], target_kind: str
    ) -> Tuple[float, float, str, bool, float]:
        """Returns (goal_x, goal_y, mode_label, should_boost, speed_scale)."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        energy = float(me["energy"])
        tick = int(obs["tick"])
        zone = obs["zone"]
        nx, ny = float(zone["next_center"][0]), float(zone["next_center"][1])
        nr = float(zone["next_radius"])
        cx, cy, cr = self._current_zone(obs)

        dist_cur_center = dist2d((mx, my), (cx, cy))
        dist_next_center = dist2d((mx, my), (nx, ny))
        enemy_mechs = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"]

        # 1. URGENT ZONE SAFETY: never take zone damage (cancels regen and ignores armor!)
        if not me["in_zone"] or dist_cur_center > max(6.0, cr - 16.0):
            sx, sy = self._quiet_safe_point(obs, nx, ny, nr)
            return sx, sy, "zone-urgent", energy > 18.0, 1.0

        time_to_shrink = float(zone["shrink_start"]) - float(obs["t"])
        if time_to_shrink < 14.0 and dist_next_center > max(12.0, nr * 0.85):
            sx, sy = self._quiet_safe_point(obs, nx, ny, nr)
            return sx, sy, "zone-rotate", energy > 45.0, 1.0

        # 2. LOW-HP / MULTI-ENEMY COVER & REGEN: break LOS when hurt or outnumbered!
        if enemy_mechs and (hp_frac < 0.58 or len(enemy_mechs) >= 2):
            primary_threat = enemy_mechs[0]
            tx, ty, tz = primary_threat["pos"]
            cover_pt = self.world.find_cover_point(mx, my, mz, tx, ty, tz)
            if cover_pt is not None and dist2d(cover_pt, (cx, cy)) < cr - 12.0:
                return cover_pt[0], cover_pt[1], "cover-regen", (hp_frac < 0.40 and energy > 35.0), 1.0
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            kx = mx - ((tx - mx) / d) * 35.0
            ky = my - ((ty - my) / d) * 35.0
            if dist2d((kx, ky), (cx, cy)) < cr - 15.0:
                return kx, ky, "kite-back", (hp_frac < 0.40 and energy > 35.0), 1.0

        # 3. SAFE SUPPLY DROPS: only contest if uncontested or we are healthy and close
        for d in obs.get("drops") or []:
            dx, dy, dz = d["pos"]
            if dist2d((dx, dy), (cx, cy)) > cr - 15.0:
                continue
            d_dist = dist2d((mx, my), (dx, dy))
            lands_in = float(d.get("lands_in", 0.0))
            holder = d.get("holder")
            enemies_at_drop = sum(1 for e in enemy_mechs if dist2d(e["pos"], (dx, dy)) < 35.0)
            if enemies_at_drop == 0 and d_dist - lands_in * 10.0 < 75.0 and (holder is None or holder == me["id"] or hp_frac > 0.70):
                return dx, dy, f"drop({lands_in:.0f}s)", (lands_in <= 1.5 and d_dist > 12.0 and energy > 45.0), 1.0

        # 4. NEARBY POWER CUBES (+4% max HP & heal, +4% dmg)
        cubes = [
            c for c in (obs.get("cubes") or [])
            if dist2d(c["pos"], (cx, cy)) < cr - 12.0 and abs(c["pos"][2] - mz) < 5.0
        ]
        if cubes:
            nearest_cube = min(cubes, key=lambda c: dist2d(c["pos"], (mx, my)))
            if dist2d(nearest_cube["pos"], (mx, my)) < 48.0:
                return nearest_cube["pos"][0], nearest_cube["pos"][1], "cube", False, 1.0

        # 5. NEARBY SECONDARY WEAPON PICKUP (when slot 1 is empty)
        weapons = me.get("weapons") or [None, None]
        if weapons[1] is None:
            good_pickups = [
                p for p in (obs.get("pickups") or [])
                if p["kind"] != "gun"
                and dist2d(p["pos"], (cx, cy)) < cr - 12.0
                and abs(p["pos"][2] - mz) < 5.0
                and tick - self.failed_pickups.get(int(p["id"]), -999) > 20
            ]
            if good_pickups:
                nearest_p = min(good_pickups, key=lambda p: dist2d(p["pos"], (mx, my)))
                if dist2d(nearest_p["pos"], (mx, my)) < 65.0:
                    return nearest_p["pos"][0], nearest_p["pos"][1], f"loot-{nearest_p['kind']}", False, 1.0

        # 6. 1v1 COMBAT POSITIONING (when healthy and engaging a single mech)
        if target is not None and target_kind == "mech":
            tx, ty, tz = target["pos"]
            d = max(1.0, dist2d((mx, my), (tx, ty)))
            held_kinds = {w["kind"] for w in weapons if w is not None}
            enemy_weapons = target.get("weapons") or []

            # Outrange enemy lightning (45m) unless we also have lightning and more HP
            if "lightning" in enemy_weapons and ("lightning" not in held_kinds or hp_frac < 0.75):
                pref_dist = 68.0
            elif "lightning" in held_kinds:
                pref_dist = 34.0
            elif "laser" in held_kinds:
                pref_dist = 75.0
            else:
                pref_dist = 58.0

            strafe_sign = 1.0 if ((tick // 16 + self.slot) % 2 == 0) else -1.0
            rx, ry = (tx - mx) / d, (ty - my) / d
            sx, sy = -ry * strafe_sign, rx * strafe_sign
            radial = clamp((d - pref_dist) / 22.0, -0.90, 0.80)

            gx = mx + (rx * radial + sx * 0.85) * 22.0
            gy = my + (ry * radial + sy * 0.85) * 22.0
            urgent_dodge = bool(target.get("charging")) and energy > 25.0
            return gx, gy, "combat-1v1", urgent_dodge, 1.0

        # 7. ROOFTOP HIGH-GROUND VANTAGE: climb a medium building inside safe zone!
        t_ground = self.world.terrain_z(mx, my)
        on_roof = (mz - t_ground) > 5.5
        if on_roof and dist_next_center <= nr * 0.88:
            return mx, my, "roof-perch", False, 0.35

        if energy >= 72.0 and self.world.rooftops:
            safe_roofs = [
                r for r in self.world.rooftops
                if dist2d(r["center"], (nx, ny)) <= nr * 0.78
                and dist2d(r["center"], (mx, my)) < 95.0
                and 6.0 <= (r["top"] - mz) <= 28.0
            ]
            if safe_roofs:
                best_roof = min(safe_roofs, key=lambda r: dist2d(r["center"], (mx, my)))
                rcx, rcy = best_roof["center"]
                self.climb_mode_until = tick + 25
                return rcx, rcy, "climb-roof", False, 1.0

        # 8. QUIET PERIMETER PATROL (use stealth speed 0.40 when near sounds so we make ZERO footstep noise)
        sx, sy = self._quiet_safe_point(obs, nx, ny, nr)
        has_near_sound = any(s.get("dist") == "near" for s in (obs.get("sounds") or []))
        speed_scale = 0.40 if (has_near_sound and dist2d((mx, my), (sx, sy)) < 35.0) else 1.0
        return sx, sy, "stealth-hold" if speed_scale < 0.5 else "safe-ring", False, speed_scale

    def _steer_move(
        self, obs: mf.Observation, goal_x: float, goal_y: float, speed_scale: float
    ) -> Tuple[List[float], bool, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        energy = float(me["energy"])
        grounded = bool(me.get("grounded", True))
        cubes = float(me.get("cubes", 0.0))
        radius = 2.5 * (1.0 + 0.05 * cubes)
        height = 9.0 * (1.0 + 0.05 * cubes)
        tick = int(obs["tick"])

        # Dodge incoming mortar shells and gun streams
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
            danger_r = 10.5 if proj.get("kind") == "mortar" else 4.8
            if d < danger_r:
                plen = math.hypot(vx, vy)
                nx_p, ny_p = -vy / plen, vx / plen
                side = 1.0 if ((mx - px) * nx_p + (my - py) * ny_p) >= 0.0 else -1.0
                dodge_x += nx_p * side * 1.8
                dodge_y += ny_p * side * 1.8

        dx = goal_x - mx + dodge_x * 14.0
        dy = goal_y - my + dodge_y * 14.0
        dist_goal = math.hypot(dx, dy)
        if dist_goal < 1.2 and dodge_x == 0.0 and dodge_y == 0.0:
            return [0.0, 0.0], False, False

        desired_yaw = math.atan2(dy, dx)

        moved = dist2d((mx, my), self.last_pos)
        self.last_pos = [mx, my, mz]
        if moved < 0.16 and dist_goal > 4.0 and speed_scale > 0.5:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)

        # If intentionally climbing a rooftop and close to building wall, jump + jet up onto roof!
        want_climb = (tick <= self.climb_mode_until and energy > 35.0)
        if want_climb:
            ux, uy = math.cos(desired_yaw), math.sin(desired_yaw)
            blk_near, top_z = self.world.is_blocked_point(mx + ux * 4.2, my + uy * 4.2, mz, radius=radius, height=height)
            if blk_near and (top_z - mz) <= 30.0:
                return [ux, uy], grounded, (not grounded)

        best_yaw = desired_yaw
        best_cost = 1e9
        need_jump = False
        need_jet = False

        offsets = [
            0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.85, -1.85, 2.4, -2.4
        ]
        if self.stuck_ticks > 5:
            bias = 1.35 if (self.slot % 2 == 0) else -1.35
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
                    if (top_z - mz) <= 18.0 and energy > 55.0 and abs(off) < 0.36:
                        can_climb = True
                    else:
                        blocked = True
                        break
                tz_ground = self.world.terrain_z(tx, ty)
                if tz_ground - mz > step_d * 1.15 + 1.2:
                    blocked = True
                    break

            if not blocked:
                cost = abs(off) + (0.45 if can_climb else 0.0)
                if cost < best_cost:
                    best_cost = cost
                    best_yaw = cand_yaw
                    if can_climb:
                        need_jump = grounded
                        need_jet = not grounded

        if self.stuck_ticks > 4 and energy > 30.0:
            need_jump = grounded
            need_jet = not grounded

        s = clamp(speed_scale, 0.0, 1.0)
        return [math.cos(best_yaw) * s, math.sin(best_yaw) * s], need_jump, need_jet

    # ------------------------------------------------------------------ Main Decision Loop
    def act(self, obs: mf.Observation) -> mf.Action:
        tick = int(obs["tick"])
        me = obs["self"]
        mx, my, mz = me["pos"]

        # Track combat & pickup events
        for ev in obs.get("events") or []:
            etype = ev.get("type")
            if etype == "damaged":
                self.last_damage_tick = tick
                if ev.get("bearing") is not None:
                    b = float(ev["bearing"])
                    self.last_attacker_bearing = b
                    self.danger_bearings.append((b, tick))
                if ev.get("by") is not None:
                    self.last_attacker_id = int(ev["by"])
            elif etype == "hit":
                self.last_dealt_tick = tick
            elif etype == "take_failed":
                self.failed_pickups[int(ev.get("id", -1))] = tick

        # Track nearby weapon sounds & beams to avoid hot zones and orient front armor
        for s in obs.get("sounds") or []:
            if s.get("dist") in ("near", "mid") and s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion"):
                b = float(s["bearing"])
                self.danger_bearings.append((b, tick))
       
