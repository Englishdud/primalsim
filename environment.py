"""Gymnasium environment wrapping PyBullet for the Primal Survival Simulation.

One instance = one training episode with a single agent.
Parallel training uses SubprocVecEnv with N_ENVS instances, each running in
its own subprocess with its own PyBullet DIRECT client.

PyBullet client creation is deferred to reset() so it is safe to pickle
the env object before SubprocVecEnv forks.
"""

import math
import random
import os
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from noise import pnoise2

from config import (
    WORLD_SIZE, WALL_HEIGHT, TERRAIN_SIZE, TERRAIN_AMPLITUDE, TERRAIN_NOISE_SCALE,
    TERRAIN_SCALE_XY, AGENT_COUNT, AGENT_MASS, AGENT_SPAWN_RADIUS,
    AGENT_SPAWN_Z_OFFSET, AGENT_COLORS, SPEAR_COUNT, SPEAR_SPAWN_RADIUS,
    SPEAR_MASS, SPEAR_HALF_LENGTH, SPEAR_RADIUS,
    SPEAR_PICKUP_RADIUS, ARMED_DAMAGE, UNARMED_DAMAGE, ATTACK_RADIUS, EAT_RADIUS,
    VISION_FOV_DEG, VISION_RANGE, VISION_RAYS,
    HIT_NOTHING, HIT_TERRAIN, HIT_MAMMOTH, HIT_AGENT,
    OBS_DIM, ACT_DIM,
    HUNGER_MAX, STAMINA_MAX,
    MAMMOTH_INITIAL_COUNT, PHYSICS_TIMESTEP, PHYSICS_SUBSTEPS, SETTLE_STEPS,
    MAX_EPISODE_STEPS,
)
from humanoid import Humanoid, PELVIS_HEIGHT_ABOVE_FEET
from mammoth import MammothManager
from rewards import compute_total_reward


class PrimalSurvivalEnv(gym.Env):
    """Single-agent gymnasium environment for the primal survival simulation.

    The environment contains one controllable agent plus MAMMOTH_INITIAL_COUNT
    mammoths and SPEAR_COUNT spears.  Other agents present in play mode are
    managed outside this class.
    """

    metadata: dict[str, Any] = {'render_modes': []}

    def __init__(self, agent_index: int = 0, seed: int | None = None):
        """Initialise spaces (PyBullet client is NOT created here – see reset).

        Args:
            agent_index: Which colour slot [0-3] this env uses.
            seed:        Optional RNG seed.
        """
        super().__init__()
        self.agent_index = agent_index
        self._seed = seed

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(ACT_DIM,), dtype=np.float32,
        )

        # State populated by reset()
        self.physics_client: int = -1
        self._terrain_id: int = -1
        self._heightmap: np.ndarray | None = None
        self._wall_ids: list[int] = []
        self._spear_ids: list[int] = []
        self._spear_held: bool = False
        self.agent: Humanoid | None = None
        self.mammoth_mgr: MammothManager | None = None
        self._step_count: int = 0

        # Reward bookkeeping
        self._first_tool_reward_given: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Tear down any existing client and start a fresh episode."""
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else (self._seed or random.randint(0, 2**31))
        random.seed(rng_seed)
        np.random.seed(rng_seed % (2**32))

        # Disconnect old client if needed
        if self.physics_client >= 0:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass

        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(PHYSICS_TIMESTEP, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            numSubSteps=PHYSICS_SUBSTEPS,
            physicsClientId=self.physics_client,
        )
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.physics_client,
        )

        # World
        self._heightmap = self._generate_heightmap()
        self._terrain_id = self._build_terrain(self._heightmap)
        self._wall_ids = self._build_walls()
        self._spear_ids = self._spawn_spears()
        self._spear_held = False

        # Mammoths
        self.mammoth_mgr = MammothManager(self.physics_client)
        for _ in range(MAMMOTH_INITIAL_COUNT):
            m = self.mammoth_mgr.spawn_random()
            # Place on terrain surface
            self._place_on_terrain(m.body_id)

        # Agent
        ax = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
        ay = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
        az = self._get_terrain_height(ax, ay) + AGENT_SPAWN_Z_OFFSET + PELVIS_HEIGHT_ABOVE_FEET + 1.0
        color = AGENT_COLORS[self.agent_index % AGENT_COUNT]
        self.agent = Humanoid(
            self.physics_client,
            [ax, ay, az],
            color,
            agent_id=self.agent_index,
        )

        # Reward state
        self._first_tool_reward_given = False
        self._step_count = 0

        # Stabilisation phase: let the body settle without policy torques
        for _ in range(SETTLE_STEPS):
            p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._build_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one step.

        Args:
            action: shape (ACT_DIM,) float32.

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.agent.update_prev_position()

        # Apply action
        discrete = self.agent.apply_action(action)

        # Handle discrete actions
        did_hit      = False
        just_killed  = False
        did_eat      = False
        first_pickup = False

        if discrete == 1:   # grab tool
            if not self.agent.is_holding_tool:
                result = self._try_grab_tool()
                if result:
                    first_pickup = not self._first_tool_reward_given
                    self._first_tool_reward_given = True

        elif discrete == 2:  # attack mammoth
            did_hit, just_killed = self._try_attack_mammoth()

        elif discrete == 3:  # eat carcass
            did_eat = self._try_eat_carcass()

        # Advance mammoth AI and physics
        agent_pos = [self.agent.get_position().tolist()]
        self.mammoth_mgr.step_all(agent_pos, self._step_count)

        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self.physics_client)

        self.agent.drain_hunger()
        self._step_count += 1

        # Build observation
        obs = self._build_observation()

        # Reward
        nearest_m, nearest_m_rel = self.mammoth_mgr.get_nearest_alive(
            self.agent.get_position()
        )
        mammoth_pos = (
            nearest_m.get_position() if nearest_m is not None else None
        )
        mammoth_visible = nearest_m is not None and self._mammoth_in_fov(nearest_m)

        reward = compute_total_reward(
            up_dot_z=self.agent.get_up_dot_z(),
            current_pos=self.agent.get_position(),
            prev_pos=self.agent.get_prev_position(),
            mammoth_pos=mammoth_pos if mammoth_visible else None,
            just_picked_up_first_tool=first_pickup,
            did_hit=did_hit,
            just_killed=just_killed,
            did_eat=did_eat,
            hunger=self.agent.hunger,
            is_high_torque=self.agent.is_high_torque_step(),
            mammoth_visible=mammoth_visible,
            is_dead=self.agent.is_dead,
        )

        terminated = self.agent.is_dead
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        info = {
            'hunger':      self.agent.hunger,
            'stamina':     self.agent.stamina,
            'holding_tool': self.agent.is_holding_tool,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Disconnect PyBullet client."""
        if self.physics_client >= 0:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = -1

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_heightmap() -> np.ndarray:
        """Create a TERRAIN_SIZE × TERRAIN_SIZE Perlin noise heightmap."""
        size = TERRAIN_SIZE
        hmap = np.zeros((size, size), dtype=np.float32)
        base = random.randint(0, 1000)
        for i in range(size):
            for j in range(size):
                hmap[i, j] = pnoise2(
                    i * TERRAIN_NOISE_SCALE + base,
                    j * TERRAIN_NOISE_SCALE + base,
                    octaves=4,
                    persistence=0.5,
                    lacunarity=2.0,
                )
        # Normalise to [0, TERRAIN_AMPLITUDE]
        hmap -= hmap.min()
        if hmap.max() > 0:
            hmap /= hmap.max()
        hmap *= TERRAIN_AMPLITUDE
        return hmap

    def _build_terrain(self, heightmap: np.ndarray) -> int:
        """Register the heightmap as a PyBullet collision heightfield."""
        size = TERRAIN_SIZE
        scale = TERRAIN_SCALE_XY
        # PyBullet heightfield expects a flat float list, row-major
        data = heightmap.flatten().tolist()
        terrain_shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[scale, scale, 1.0],
            heightfieldData=data,
            numHeightfieldRows=size,
            numHeightfieldColumns=size,
            replaceHeightfieldIndex=-1,
            physicsClientId=self.physics_client,
        )
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            terrain_id, -1,
            lateralFriction=0.8,
            physicsClientId=self.physics_client,
        )
        return terrain_id

    def _build_walls(self) -> list[int]:
        """Create 4 static infinite planes at the sandbox boundaries."""
        half = WORLD_SIZE / 2
        # (planeNormal, position)
        wall_defs = [
            ([  1,  0,  0], [-half,  0,     0]),   # west wall, normal +x
            ([ -1,  0,  0], [ half,  0,     0]),   # east wall, normal -x
            ([  0,  1,  0], [   0, -half,   0]),   # south wall, normal +y
            ([  0, -1,  0], [   0,  half,   0]),   # north wall, normal -y
        ]
        ids = []
        for normal, pos in wall_defs:
            shape = p.createCollisionShape(
                p.GEOM_PLANE,
                planeNormal=normal,
                physicsClientId=self.physics_client,
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                basePosition=pos,
                physicsClientId=self.physics_client,
            )
            ids.append(wall_id)
        return ids

    def _spawn_spears(self) -> list[int]:
        """Place SPEAR_COUNT thin cylinder bodies on the terrain."""
        ids = []
        shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=SPEAR_RADIUS,
            height=SPEAR_HALF_LENGTH * 2,
            physicsClientId=self.physics_client,
        )
        for _ in range(SPEAR_COUNT):
            x = random.uniform(-SPEAR_SPAWN_RADIUS, SPEAR_SPAWN_RADIUS)
            y = random.uniform(-SPEAR_SPAWN_RADIUS, SPEAR_SPAWN_RADIUS)
            z = self._get_terrain_height(x, y) + SPEAR_HALF_LENGTH + 0.02
            spear_id = p.createMultiBody(
                baseMass=SPEAR_MASS,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=-1,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
                physicsClientId=self.physics_client,
            )
            p.changeDynamics(
                spear_id, -1,
                lateralFriction=0.5,
                physicsClientId=self.physics_client,
            )
            ids.append(spear_id)
        return ids

    def _place_on_terrain(self, body_id: int) -> None:
        """Teleport a body so it rests on the terrain surface."""
        pos, ori = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self.physics_client
        )
        # Use the fixed height function + 5.0 units of air
        target_z = self._get_terrain_height(pos[0], pos[1]) + 5.0
        
        p.resetBasePositionAndOrientation(
            body_id, 
            [pos[0], pos[1], target_z], 
            ori,
            physicsClientId=self.physics_client,
        )
    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Assemble the 75-float observation vector for the controlled agent."""
        angles, vels = self.agent.get_joint_obs()            # (12,), (12,)
        quat, lin_vel, ang_vel = self.agent.get_base_obs()   # (4,), (3,), (3,)
        vision = self._cast_vision_rays()                    # (32,)

        agent_pos = self.agent.get_position()

        nearest_m, nearest_m_rel = self.mammoth_mgr.get_nearest_alive(agent_pos)
        mammoth_rel = nearest_m_rel if nearest_m is not None else np.zeros(3, np.float32)

        nearest_tool_rel = self._nearest_spear_rel(agent_pos)

        hunger_norm  = float(self.agent.hunger)  / HUNGER_MAX
        stamina_norm = float(self.agent.stamina) / STAMINA_MAX
        tool_flag    = float(self.agent.is_holding_tool)

        obs = np.concatenate([
            angles,          # 0:12
            vels,            # 12:24
            quat,            # 24:28
            lin_vel,         # 28:31
            ang_vel,         # 31:34
            vision,          # 34:66
            mammoth_rel,     # 66:69
            nearest_tool_rel,# 69:72
            [hunger_norm, stamina_norm, tool_flag],  # 72:75
        ]).astype(np.float32)
        return obs

    def _cast_vision_rays(self) -> np.ndarray:
        """Cast VISION_RAYS raycasts in a forward-facing fan.

        Returns a (32,) array: alternating (normalised_distance, hit_type).
        """
        head_pos = self.agent.get_head_position()
        head_ori = self.agent.get_head_orientation()   # (x,y,z,w)

        mat = p.getMatrixFromQuaternion(head_ori)
        # Forward = local Y axis column in Panda3D / PyBullet convention
        forward = np.array([mat[1], mat[4], mat[7]], dtype=np.float64)
        # Keep forward horizontal
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0.0, 1.0, 0.0])
        else:
            forward /= norm

        right = np.array([-forward[1], forward[0], 0.0])

        half_fov = math.radians(VISION_FOV_DEG / 2)
        from_pts = []
        to_pts   = []
        for i in range(VISION_RAYS):
            t = i / max(VISION_RAYS - 1, 1)
            angle = -half_fov + t * (2 * half_fov)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dir_ray = (cos_a * forward + sin_a * right) * VISION_RANGE
            from_pts.append([head_pos[0], head_pos[1], head_pos[2]])
            to_pts.append([
                head_pos[0] + dir_ray[0],
                head_pos[1] + dir_ray[1],
                head_pos[2] + dir_ray[2],
            ])

        results = p.rayTestBatch(
            from_pts, to_pts,
            physicsClientId=self.physics_client,
        )

        mammoth_ids = set(self.mammoth_mgr.all_body_ids())
        out = np.zeros(VISION_RAYS * 2, dtype=np.float32)

        for i, res in enumerate(results):
            hit_id      = res[0]
            hit_fraction = float(res[2])
            dist_norm    = hit_fraction  # 0=at agent, 1=max range

            if hit_id < 0:
                hit_type = HIT_NOTHING
            elif hit_id == self._terrain_id:
                hit_type = HIT_TERRAIN
            elif hit_id in mammoth_ids:
                hit_type = HIT_MAMMOTH
            elif hit_id == self.agent.body_id:
                hit_type = HIT_NOTHING  # ignore own body
            else:
                hit_type = HIT_AGENT

            out[i * 2]     = dist_norm
            out[i * 2 + 1] = hit_type

        return out

    def _nearest_spear_rel(self, agent_pos: np.ndarray) -> np.ndarray:
        """Relative position (3-D) to the nearest uncollected spear."""
        if self.agent.is_holding_tool or not self._spear_ids:
            return np.zeros(3, dtype=np.float32)

        best_dist = float('inf')
        best_pos  = None
        for sid in self._spear_ids:
            try:
                spos, _ = p.getBasePositionAndOrientation(
                    sid, physicsClientId=self.physics_client
                )
            except Exception:
                continue
            d = np.linalg.norm(np.array(spos) - agent_pos)
            if d < best_dist:
                best_dist = d
                best_pos  = np.array(spos, dtype=np.float32)

        if best_pos is None:
            return np.zeros(3, dtype=np.float32)
        return best_pos - agent_pos

    # ------------------------------------------------------------------
    # Discrete-action handlers
    # ------------------------------------------------------------------

    def _try_grab_tool(self) -> bool:
        """Attempt to grab the nearest spear within SPEAR_PICKUP_RADIUS."""
        agent_pos = self.agent.get_position()
        for sid in self._spear_ids:
            try:
                spos, _ = p.getBasePositionAndOrientation(
                    sid, physicsClientId=self.physics_client
                )
            except Exception:
                continue
            if np.linalg.norm(np.array(spos) - agent_pos) <= SPEAR_PICKUP_RADIUS:
                # Remove from world, mark held
                p.removeBody(sid, physicsClientId=self.physics_client)
                self._spear_ids.remove(sid)
                self.agent.grab_tool()
                return True
        return False

    def _try_attack_mammoth(self) -> tuple[bool, bool]:
        """Attack the nearest mammoth within ATTACK_RADIUS.

        Returns (hit, killed).
        """
        agent_pos = self.agent.get_position()
        damage = ARMED_DAMAGE if self.agent.is_holding_tool else UNARMED_DAMAGE
        for m in self.mammoth_mgr.mammoths:
            if m.is_dead:
                continue
            if np.linalg.norm(m.get_position() - agent_pos) <= ATTACK_RADIUS:
                killed = m.take_damage(damage)
                return True, killed
        return False, False

    def _try_eat_carcass(self) -> bool:
        """Eat the nearest carcass within EAT_RADIUS."""
        agent_pos = self.agent.get_position()
        carcass, dist = self.mammoth_mgr.get_nearest_carcass(agent_pos)
        if carcass is not None and dist <= EAT_RADIUS:
            self.agent.eat()
            return True
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_terrain_height(self, x: float, y: float) -> float:
        """Query terrain height at (x, y) via a downward raycast."""
        # We cast from 100 down to -50
        result = p.rayTest(
            [x, y, 100.0],
            [x, y, -50.0],
            physicsClientId=self.physics_client,
        )
        
        if result:
            for hit in result:
                hit_id = hit[0]
                # ONLY return the height if we hit the terrain ID
                if hit_id == self._terrain_id:
                    return float(hit[3][2])
        
        # Fallback to a safe middle-ground height if ray fails
        return 5.0

    def _mammoth_in_fov(self, m: Any) -> bool:
        """True if the mammoth is within the agent's vision cone."""
        agent_pos = self.agent.get_position()
        m_pos     = m.get_position()
        delta     = m_pos - agent_pos
        dist      = float(np.linalg.norm(delta))
        if dist > VISION_RANGE or dist < 1e-3:
            return False

        # Get agent forward vector
        head_ori = self.agent.get_head_orientation()
        mat      = p.getMatrixFromQuaternion(head_ori)
        forward  = np.array([mat[1], mat[4], mat[7]])
        forward[2] = 0.0
        fn = np.linalg.norm(forward)
        if fn < 1e-6:
            return True
        forward /= fn

        delta_h = delta.copy()
        delta_h[2] = 0.0
        dn = np.linalg.norm(delta_h)
        if dn < 1e-6:
            return True
        delta_h /= dn

        cos_angle = float(np.dot(forward, delta_h))
        half_fov_cos = math.cos(math.radians(VISION_FOV_DEG / 2))
        return cos_angle >= half_fov_cos
