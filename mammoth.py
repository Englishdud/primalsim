"""Mammoth entity – PyBullet physics body, scripted AI, and Panda3D visuals.

Mammoths use simple velocity-PD steering toward waypoints, with avoidance
steering when agents are nearby.  No RL is involved.
"""

import math
import random
import time
import numpy as np
import pybullet as p

from config import (
    MAMMOTH_MASS, MAMMOTH_SPEED, MAMMOTH_HP,
    MAMMOTH_LINEAR_DAMPING, MAMMOTH_ANGULAR_DAMPING,
    MAMMOTH_AVOIDANCE_DIST, MAMMOTH_AVOIDANCE_RECALC_STEPS,
    MAMMOTH_WAYPOINT_INTERVAL, MAMMOTH_CARCASS_LIFETIME,
    MAMMOTH_BODY_HALF_EXTENTS, MAMMOTH_LEG_HALF_EXTENTS, MAMMOTH_TUSK_HALF_EXTENTS,
    MAMMOTH_SPAWN_RADIUS, MAMMOTH_PD_KP, MAMMOTH_PD_KD, MAMMOTH_MAX_FORCE,
    PHYSICS_TIMESTEP,
    WORLD_SIZE,
)


# Colours (RGBA)
_COLOR_ALIVE  = (0.35, 0.25, 0.18, 1.0)   # dark brown
_COLOR_FLASH  = (1.00, 0.15, 0.15, 1.0)   # damage-flash red
_COLOR_DEAD   = (0.25, 0.25, 0.25, 1.0)   # grey carcass
_FLASH_FRAMES = 12   # ~0.2 s at 60 fps


class Mammoth:
    """A single mammoth with physics, scripted AI, and optional Panda3D visuals."""

    def __init__(
        self,
        physics_client: int,
        start_pos: list,
        render_parent=None,   # Panda3D NodePath or None
    ):
        """Create the mammoth PyBullet body and optional Panda3D nodes.

        Args:
            physics_client: PyBullet physics client ID.
            start_pos:      [x, y, z] initial world position.
            render_parent:  Panda3D NodePath for visuals, or None.
        """
        self.client = physics_client
        self.hp: float = MAMMOTH_HP
        self.is_dead: bool = False
        self._carcass_timer: float = 0.0      # seconds elapsed since death
        self._waypoint: np.ndarray = np.array(start_pos[:2], dtype=np.float32)
        self._ai_counter: int = 0
        self._waypoint_timer: float = 0.0

        # Flash state
        self._flash_frames_left: int = 0

        # Build physics
        self.body_id: int = self._build_body(start_pos)

        # Panda3D
        self._vis_root = None
        self._vis_nodes: dict = {}
        if render_parent is not None:
            self._build_visuals(render_parent)

    # ------------------------------------------------------------------
    # PyBullet body
    # ------------------------------------------------------------------

    def _build_body(self, start_pos: list) -> int:
        """Create the mammoth collision body in PyBullet."""
        bx, by, bz = MAMMOTH_BODY_HALF_EXTENTS
        shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[bx, by, bz],
            physicsClientId=self.client,
        )
        body_id = p.createMultiBody(
            baseMass=MAMMOTH_MASS,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=-1,
            basePosition=[start_pos[0], start_pos[1],
                          start_pos[2] + bz],   # raise so bottom is at ground
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.client,
        )
        p.changeDynamics(
            body_id, -1,
            linearDamping=MAMMOTH_LINEAR_DAMPING,
            angularDamping=MAMMOTH_ANGULAR_DAMPING,
            lateralFriction=0.8,
            physicsClientId=self.client,
        )
        return body_id

    # ------------------------------------------------------------------
    # Panda3D visuals
    # ------------------------------------------------------------------

    def _build_visuals(self, parent_np) -> None:
        """Build a simple box-based mammoth from Panda3D geometry primitives."""
        from play_utils import make_box_np

        bx, by, bz = MAMMOTH_BODY_HALF_EXTENTS
        lx, ly, lz = MAMMOTH_LEG_HALF_EXTENTS
        tx, ty, tz = MAMMOTH_TUSK_HALF_EXTENTS

        self._vis_root = parent_np.attachNewNode('mammoth_root')

        # Main body
        body_np = make_box_np(bx, by, bz, _COLOR_ALIVE)
        body_np.reparentTo(self._vis_root)
        self._vis_nodes['body'] = body_np

        # 4 legs (visual only, static offsets)
        leg_offsets = [
            (-bx + lx, -by + ly,  -(bz + lz)),
            (-bx + lx,  by - ly,  -(bz + lz)),
            ( bx - lx, -by + ly,  -(bz + lz)),
            ( bx - lx,  by - ly,  -(bz + lz)),
        ]
        for i, (ox, oy, oz) in enumerate(leg_offsets):
            leg = make_box_np(lx, ly, lz, (_COLOR_ALIVE[0]*0.8,
                                            _COLOR_ALIVE[1]*0.8,
                                            _COLOR_ALIVE[2]*0.8, 1.0))
            leg.reparentTo(self._vis_root)
            leg.setPos(ox, oy, oz)
            self._vis_nodes[f'leg_{i}'] = leg

        # 2 tusks at the front
        for i, side in enumerate((-1, 1)):
            tusk = make_box_np(tx, ty, tz, (0.85, 0.82, 0.75, 1.0))
            tusk.reparentTo(self._vis_root)
            tusk.setPos(side * (bx * 0.5), by + ty, bz * 0.4)
            self._vis_nodes[f'tusk_{i}'] = tusk

    # ------------------------------------------------------------------
    # AI step
    # ------------------------------------------------------------------

    def step_ai(
        self,
        agent_positions: list,
        step_num: int,
        dt: float = PHYSICS_TIMESTEP,
    ) -> bool:
        """Advance mammoth scripted behaviour by one physics step.

        Returns False when the carcass has fully expired and should be removed.
        """
        if self.is_dead:
            self._carcass_timer += dt
            return self._carcass_timer < MAMMOTH_CARCASS_LIFETIME

        self._waypoint_timer += dt
        self._ai_counter += 1

        # Recalculate steering every N steps
        if self._ai_counter % MAMMOTH_AVOIDANCE_RECALC_STEPS == 0:
            self._recalculate_waypoint(agent_positions)

        # Periodically pick a new random waypoint
        if self._waypoint_timer >= MAMMOTH_WAYPOINT_INTERVAL:
            self._pick_random_waypoint()
            self._waypoint_timer = 0.0

        # Apply PD force toward waypoint (2-D, ignore z)
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        vel, _ = p.getBaseVelocity(
            self.body_id, physicsClientId=self.client
        )

        curr_xy = np.array([pos[0], pos[1]])
        curr_vel_xy = np.array([vel[0], vel[1]])
        wp = self._waypoint

        direction = wp - curr_xy
        dist = np.linalg.norm(direction)
        if dist > 0.5:
            target_vel = (direction / dist) * MAMMOTH_SPEED
            force_xy = (
                MAMMOTH_PD_KP * (target_vel - curr_vel_xy)
                - MAMMOTH_PD_KD * curr_vel_xy
            )
            mag = np.linalg.norm(force_xy)
            if mag > MAMMOTH_MAX_FORCE:
                force_xy = force_xy / mag * MAMMOTH_MAX_FORCE
            p.applyExternalForce(
                self.body_id, -1,
                [float(force_xy[0]), float(force_xy[1]), 0.0],
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        # Face the direction of travel
        if dist > 1.0:
            self._face_waypoint(pos, wp)

        return True

    def _recalculate_waypoint(self, agent_positions: list) -> None:
        """Steer away from the nearest agent if within avoidance radius."""
        if not agent_positions:
            return
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        my_pos = np.array([pos[0], pos[1]])

        nearest_dist = float('inf')
        nearest_agent = None
        for ap in agent_positions:
            d = np.linalg.norm(np.array([ap[0], ap[1]]) - my_pos)
            if d < nearest_dist:
                nearest_dist = d
                nearest_agent = np.array([ap[0], ap[1]])

        if nearest_agent is not None and nearest_dist < MAMMOTH_AVOIDANCE_DIST:
            # Flee in opposite direction
            away = my_pos - nearest_agent
            norm = np.linalg.norm(away)
            if norm > 0.01:
                away = away / norm
            flee_target = my_pos + away * (MAMMOTH_AVOIDANCE_DIST * 1.2)
            # Clamp to world
            half = WORLD_SIZE / 2 - 20
            flee_target = np.clip(flee_target, -half, half)
            self._waypoint = flee_target

    def _pick_random_waypoint(self) -> None:
        """Choose a new random point inside the sandbox."""
        half = WORLD_SIZE / 2 - 30
        self._waypoint = np.array([
            random.uniform(-half, half),
            random.uniform(-half, half),
        ], dtype=np.float32)

    def _face_waypoint(self, pos: tuple, wp: np.ndarray) -> None:
        """Rotate the mammoth body to face its waypoint."""
        dx = float(wp[0]) - float(pos[0])
        dy = float(wp[1]) - float(pos[1])
        yaw = math.atan2(dx, dy)
        ori = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self.body_id,
            [pos[0], pos[1], pos[2]],
            ori,
            physicsClientId=self.client,
        )

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def take_damage(self, damage: float) -> bool:
        """Subtract HP. Returns True if this hit killed the mammoth."""
        if self.is_dead:
            return False
        self.hp -= damage
        self._flash_frames_left = _FLASH_FRAMES
        if self.hp <= 0.0:
            self.hp = 0.0
            self.is_dead = True
            # Make carcass static
            p.changeDynamics(
                self.body_id, -1,
                mass=0.0,
                physicsClientId=self.client,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self) -> np.ndarray:
        """World position of the mammoth body centre."""
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        return np.array(pos, dtype=np.float32)

    def is_carcass_expired(self) -> bool:
        """True when the carcass has exceeded its lifetime."""
        return self.is_dead and (self._carcass_timer >= MAMMOTH_CARCASS_LIFETIME)

    # ------------------------------------------------------------------
    # Panda3D visual sync
    # ------------------------------------------------------------------

    def sync_visuals(self) -> None:
        """Update Panda3D node transforms from PyBullet state."""
        if self._vis_root is None:
            return

        from humanoid import _set_np_transform

        pos, ori = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        _set_np_transform(self._vis_root, pos, ori)

        # Damage flash
        if self._flash_frames_left > 0:
            self._flash_frames_left -= 1
            t = self._flash_frames_left / _FLASH_FRAMES
            # Lerp from flash colour back to alive/dead colour
            base = _COLOR_DEAD if self.is_dead else _COLOR_ALIVE
            r = _COLOR_FLASH[0] * t + base[0] * (1 - t)
            g = _COLOR_FLASH[1] * t + base[1] * (1 - t)
            b = _COLOR_FLASH[2] * t + base[2] * (1 - t)
            self._set_body_color((r, g, b, 1.0))
        elif self.is_dead:
            # Fade to grey over carcass lifetime
            alpha = max(0.0, 1.0 - self._carcass_timer / MAMMOTH_CARCASS_LIFETIME)
            self._set_body_color((_COLOR_DEAD[0], _COLOR_DEAD[1],
                                   _COLOR_DEAD[2], alpha))

    def _set_body_color(self, rgba: tuple) -> None:
        """Apply a flat colour to the main body node."""
        if 'body' in self._vis_nodes:
            self._vis_nodes['body'].setColor(*rgba)

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove PyBullet body and detach Panda3D nodes."""
        try:
            p.removeBody(self.body_id, physicsClientId=self.client)
        except Exception:
            pass
        if self._vis_root is not None:
            try:
                self._vis_root.removeNode()
            except Exception:
                pass
        self._vis_nodes.clear()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MammothManager:
    """Owns and steers all mammoth instances in a single physics world."""

    def __init__(
        self,
        physics_client: int,
        render_parent=None,
    ):
        """Set up the manager (does not spawn any mammoths yet).

        Args:
            physics_client: PyBullet client ID shared by all mammoths.
            render_parent:  Panda3D NodePath for visuals, or None.
        """
        self.client = physics_client
        self._render_parent = render_parent
        self.mammoths: list[Mammoth] = []

    def spawn(self, pos: list) -> Mammoth:
        """Spawn a single mammoth at *pos* = [x, y, z]."""
        m = Mammoth(self.client, pos, render_parent=self._render_parent)
        self.mammoths.append(m)
        return m

    def spawn_random(self) -> Mammoth:
        """Spawn a mammoth at a random location inside the sandbox."""
        half = WORLD_SIZE / 2 - 50
        x = random.uniform(-half, half)
        y = random.uniform(-half, half)
        return self.spawn([x, y, 0.0])   # z is adjusted in _build_body

    def step_all(
        self,
        agent_positions: list,
        step_num: int,
        dt: float = PHYSICS_TIMESTEP,
    ) -> None:
        """Advance AI and expire old carcasses.  Must be called every step."""
        to_remove = []
        for m in self.mammoths:
            alive = m.step_ai(agent_positions, step_num, dt)
            if not alive:
                to_remove.append(m)
        for m in to_remove:
            m.remove()
            self.mammoths.remove(m)

    def sync_all_visuals(self) -> None:
        """Update all Panda3D node transforms."""
        for m in self.mammoths:
            m.sync_visuals()

    def get_nearest_alive(
        self, from_pos: np.ndarray
    ) -> tuple['Mammoth | None', np.ndarray]:
        """Return (mammoth, relative_position_3d) for the nearest living mammoth.

        Returns (None, zeros(3)) when no living mammoths exist.
        """
        best_m = None
        best_dist = float('inf')
        for m in self.mammoths:
            if m.is_dead:
                continue
            d = np.linalg.norm(m.get_position() - from_pos)
            if d < best_dist:
                best_dist = d
                best_m = m
        if best_m is None:
            return None, np.zeros(3, dtype=np.float32)
        rel = best_m.get_position() - from_pos
        return best_m, rel.astype(np.float32)

    def get_nearest_carcass(
        self, from_pos: np.ndarray
    ) -> tuple['Mammoth | None', float]:
        """Return (carcass, distance) for the nearest dead mammoth."""
        best_m = None
        best_dist = float('inf')
        for m in self.mammoths:
            if not m.is_dead:
                continue
            d = float(np.linalg.norm(m.get_position() - from_pos))
            if d < best_dist:
                best_dist = d
                best_m = m
        return best_m, best_dist

    def alive_body_ids(self) -> list[int]:
        """PyBullet body IDs for all living mammoths (used for vision ray checks)."""
        return [m.body_id for m in self.mammoths if not m.is_dead]

    def all_body_ids(self) -> list[int]:
        """PyBullet body IDs for all mammoths including carcasses."""
        return [m.body_id for m in self.mammoths]

    def alive_positions(self) -> list[np.ndarray]:
        """World positions of all living mammoths."""
        return [m.get_position() for m in self.mammoths if not m.is_dead]

    def count_alive(self) -> int:
        """Number of living (non-dead) mammoths."""
        return sum(1 for m in self.mammoths if not m.is_dead)

    def clear(self) -> None:
        """Remove all mammoths."""
        for m in self.mammoths:
            m.remove()
        self.mammoths.clear()
