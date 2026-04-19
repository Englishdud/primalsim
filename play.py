"""Visual play mode for the Primal Survival Simulation.

Combines a PyBullet DIRECT physics server (for all collision and joint
simulation) with a Panda3D renderer (for all visuals).  Every frame:

  1. All 4 agents query the shared policy for actions.
  2. Actions are applied to PyBullet joints / discrete handlers.
  3. Mammoth AI steps.
  4. Physics substeps run.
  5. All Panda3D NodePaths are synced to PyBullet positions.
  6. HUD updates.
  7. Camera lerps toward the selected agent.

Usage:
    python play.py
    python play.py --checkpoint ./checkpoints/latest.zip
    python play.py --random   (random policy, no checkpoint needed)
"""

import math
import random
import time
import argparse
import os

import numpy as np
import pybullet as p
import pybullet_data

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, DirectionalLight,
    Vec3, Vec4, NodePath,
    TransparencyAttrib, CardMaker,
    LPoint3f, LQuaternionf,
    ClockObject,
)

from config import (
    WORLD_SIZE, WALL_HEIGHT, TERRAIN_SIZE, TERRAIN_AMPLITUDE,
    TERRAIN_NOISE_SCALE, TERRAIN_SCALE_XY,
    AGENT_COUNT, AGENT_COLORS, AGENT_SPAWN_RADIUS, AGENT_SPAWN_Z_OFFSET,
    SPEAR_COUNT, SPEAR_SPAWN_RADIUS, SPEAR_MASS, SPEAR_HALF_LENGTH, SPEAR_RADIUS,
    MAMMOTH_INITIAL_COUNT,
    PHYSICS_TIMESTEP, PHYSICS_SUBSTEPS, SETTLE_STEPS,
    CAMERA_DISTANCE, CAMERA_HEIGHT, CAMERA_LERP,
    ATTACK_RADIUS, SPEAR_PICKUP_RADIUS, EAT_RADIUS,
    ARMED_DAMAGE, UNARMED_DAMAGE,
    HUNGER_MAX, STAMINA_MAX,
    OBS_DIM, ACT_DIM,
    VISION_FOV_DEG, VISION_RANGE, VISION_RAYS,
    HIT_NOTHING, HIT_TERRAIN, HIT_MAMMOTH, HIT_AGENT,
)
from humanoid import Humanoid, PELVIS_HEIGHT_ABOVE_FEET, _set_np_transform
from mammoth import MammothManager
from hud import HUD
from sandbox_controls import SandboxControls
from play_utils import make_box_np, make_cylinder_np, build_terrain_mesh
from environment import PrimalSurvivalEnv   # reuse terrain/heightmap helpers


class PrimalSimApp(ShowBase):
    """Panda3D ShowBase application for the visual simulation mode."""

    def __init__(self, policy=None):
        """Build the full simulation scene and start the main loop.

        Args:
            policy: A loaded SB3 PPO model, or None for random actions.
        """
        ShowBase.__init__(self)
        self.disableMouse()

        self.policy = policy

        # Deep blue sky
        self.win.setClearColor(Vec4(0.08, 0.15, 0.35, 1.0))

        # Deterministic seed for this session
        random.seed(42)
        np.random.seed(42)

        # State
        self.selected_agent: int = 0
        self._sim_time: float    = 0.0
        self._step_num: int      = 0

        # Camera lerp state
        self._cam_pos   = Vec3(0, -CAMERA_DISTANCE, CAMERA_HEIGHT)
        self._cam_look  = Vec3(0, 0, 0)

        # ── Physics ────────────────────────────────────────────────
        self._physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setTimeStep(PHYSICS_TIMESTEP, physicsClientId=self._physics_client)
        p.setPhysicsEngineParameter(
            numSubSteps=PHYSICS_SUBSTEPS,
            physicsClientId=self._physics_client,
        )
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._physics_client,
        )

        # ── World ──────────────────────────────────────────────────
        self._heightmap  = PrimalSurvivalEnv._generate_heightmap()
        self._terrain_id = self._build_terrain_physics(self._heightmap)
        self._wall_ids   = self._build_walls_physics()
        self._spear_ids: list[int]   = []
        self._spear_nodes: list[NodePath] = []

        # Lighting
        self._setup_lighting()

        # Terrain visual
        terrain_np = build_terrain_mesh(self._heightmap, WORLD_SIZE)
        terrain_np.reparentTo(self.render)

        # Wall visuals
        self._build_wall_visuals()

        # Mammoths
        self.mammoth_mgr = MammothManager(
            self._physics_client, render_parent=self.render
        )

        # Agents
        self.agents: list[Humanoid] = []

        # Spears
        self._spawn_spears()

        # Build everything
        self._spawn_all_agents()
        self._spawn_initial_mammoths()
        self._stabilise()

        # HUD and controls
        self.hud      = HUD(self)
        self.controls = SandboxControls(self)

        # Main update task
        self.taskMgr.add(self._update, 'primal_sim_update')

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _build_terrain_physics(self, heightmap: np.ndarray) -> int:
        """Register the heightmap as a PyBullet heightfield."""
        size  = TERRAIN_SIZE
        scale = TERRAIN_SCALE_XY
        data  = heightmap.flatten().tolist()
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[scale, scale, 1.0],
            heightfieldData=data,
            numHeightfieldRows=size,
            numHeightfieldColumns=size,
            replaceHeightfieldIndex=-1,
            physicsClientId=self._physics_client,
        )
        tid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=[0, 0, 0],
            physicsClientId=self._physics_client,
        )
        p.changeDynamics(tid, -1, lateralFriction=0.8,
                         physicsClientId=self._physics_client)
        return tid

    def _build_walls_physics(self) -> list[int]:
        """Four static infinite planes at the sandbox boundaries."""
        half = WORLD_SIZE / 2
        defs = [
            ([ 1,  0, 0], [-half,    0, 0]),
            ([-1,  0, 0], [ half,    0, 0]),
            ([ 0,  1, 0], [   0, -half, 0]),
            ([ 0, -1, 0], [   0,  half, 0]),
        ]
        ids = []
        for normal, pos in defs:
            shape = p.createCollisionShape(
                p.GEOM_PLANE, planeNormal=normal,
                physicsClientId=self._physics_client,
            )
            wid = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=shape,
                basePosition=pos,
                physicsClientId=self._physics_client,
            )
            ids.append(wid)
        return ids

    def _build_wall_visuals(self) -> None:
        """Translucent card planes marking the sandbox boundaries."""
        half = WORLD_SIZE / 2
        wall_configs = [
            # (pos, hpr, width, height)
            ((-half,    0, WALL_HEIGHT/2), (90,  0, 0), WORLD_SIZE, WALL_HEIGHT),
            (( half,    0, WALL_HEIGHT/2), (90,  0, 0), WORLD_SIZE, WALL_HEIGHT),
            ((   0, -half, WALL_HEIGHT/2), ( 0,  0, 0), WORLD_SIZE, WALL_HEIGHT),
            ((   0,  half, WALL_HEIGHT/2), ( 0,  0, 0), WORLD_SIZE, WALL_HEIGHT),
        ]
        for (px, py, pz), (h, pr, r), w, ht in wall_configs:
            cm = CardMaker('wall')
            cm.setFrame(-w/2, w/2, -ht/2, ht/2)
            np_node = self.render.attachNewNode(cm.generate())
            np_node.setPos(px, py, pz)
            np_node.setHpr(h, pr, r)
            np_node.setTransparency(TransparencyAttrib.MAlpha)
            np_node.setColor(0.6, 0.7, 0.9, 0.18)
            np_node.setTwoSided(True)

    def _setup_lighting(self) -> None:
        """Add sun directional light + ambient fill."""
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(0.90, 0.88, 0.80, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)

        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.28, 0.28, 0.32, 1.0))
        amb_np = self.render.attachNewNode(ambient)
        self.render.setLight(amb_np)

    def _spawn_spears(self) -> None:
        """Create SPEAR_COUNT spear physics bodies + orange cylinder visuals."""
        shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=SPEAR_RADIUS,
            height=SPEAR_HALF_LENGTH * 2,
            physicsClientId=self._physics_client,
        )
        for _ in range(SPEAR_COUNT):
            x = random.uniform(-SPEAR_SPAWN_RADIUS, SPEAR_SPAWN_RADIUS)
            y = random.uniform(-SPEAR_SPAWN_RADIUS, SPEAR_SPAWN_RADIUS)
            z = self._get_terrain_height(x, y) + SPEAR_HALF_LENGTH + 0.05
            import math as _math
            spear_id = p.createMultiBody(
                baseMass=SPEAR_MASS,
                baseCollisionShapeIndex=shape,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler(
                    [_math.pi / 2, 0, 0]
                ),
                physicsClientId=self._physics_client,
            )
            self._spear_ids.append(spear_id)
            # Visual
            cyl = make_cylinder_np(SPEAR_RADIUS, SPEAR_HALF_LENGTH,
                                    (1.0, 0.45, 0.05, 1.0))
            cyl.reparentTo(self.render)
            self._spear_nodes.append(cyl)

    def _spawn_all_agents(self) -> None:
        """Spawn the 4 agent Humanoids with Panda3D visuals."""
        for i in range(AGENT_COUNT):
            x = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
            y = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
            z = (self._get_terrain_height(x, y)
                 + AGENT_SPAWN_Z_OFFSET + PELVIS_HEIGHT_ABOVE_FEET)
            agent = Humanoid(
                self._physics_client,
                [x, y, z],
                AGENT_COLORS[i],
                agent_id=i,
                render_parent=self.render,
            )
            self.agents.append(agent)

    def _spawn_initial_mammoths(self) -> None:
        """Spawn starting mammoths on the terrain surface."""
        for _ in range(MAMMOTH_INITIAL_COUNT):
            m = self.mammoth_mgr.spawn_random()
            pos = m.get_position()
            z = self._get_terrain_height(pos[0], pos[1]) + 2.0
            p.resetBasePositionAndOrientation(
                m.body_id, [pos[0], pos[1], z], (0, 0, 0, 1),
                physicsClientId=self._physics_client,
            )

    def _stabilise(self) -> None:
        """Run settle steps so agents stand before policy takes over."""
        for _ in range(SETTLE_STEPS):
            p.stepSimulation(physicsClientId=self._physics_client)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def _update(self, task):
        """Per-frame update: physics → policy → visuals → HUD → camera."""
        dt = self.taskMgr.globalClock.getDt()
        self._sim_time += dt

        # 1. Get observations for all 4 agents
        obs_batch = self._get_all_observations()   # (4, OBS_DIM)

        # 2. Get actions from policy (or random)
        actions = self._predict_actions(obs_batch)  # (4, ACT_DIM)

        # 3. Apply actions
        for i, agent in enumerate(self.agents):
            if agent.is_dead:
                continue
            agent.update_prev_position()
            discrete = agent.apply_action(actions[i])
            self._handle_discrete(agent, i, discrete)

        # 4. Mammoth AI
        alive_positions = [a.get_position().tolist()
                           for a in self.agents if not a.is_dead]
        self.mammoth_mgr.step_all(alive_positions, self._step_num)

        # 5. Physics substeps
        for _ in range(PHYSICS_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._physics_client)

        # 6. Drain hunger
        for agent in self.agents:
            agent.drain_hunger()

        self._step_num += 1

        # 7. Sync visuals
        for agent in self.agents:
            agent.sync_visuals()
        self.mammoth_mgr.sync_all_visuals()
        self._sync_spear_visuals()

        # 8. Camera
        self._update_camera()

        # 9. HUD
        self._update_hud()

        return task.cont

    # ------------------------------------------------------------------
    # Observation / action helpers
    # ------------------------------------------------------------------

    def _get_all_observations(self) -> np.ndarray:
        """Build a (4, OBS_DIM) observation batch for all agents."""
        batch = np.zeros((AGENT_COUNT, OBS_DIM), dtype=np.float32)
        mammoth_ids = set(self.mammoth_mgr.all_body_ids())
        for i, agent in enumerate(self.agents):
            if agent.is_dead:
                continue
            batch[i] = self._build_agent_obs(agent, mammoth_ids)
        return batch

    def _build_agent_obs(
        self, agent: Humanoid, mammoth_ids: set
    ) -> np.ndarray:
        """Assemble the 75-float observation for one agent."""
        angles, vels = agent.get_joint_obs()
        quat, lin_vel, ang_vel = agent.get_base_obs()
        vision = self._cast_vision_rays(agent, mammoth_ids)

        agent_pos = agent.get_position()
        nearest_m, nearest_m_rel = self.mammoth_mgr.get_nearest_alive(agent_pos)
        mammoth_rel = nearest_m_rel if nearest_m is not None else np.zeros(3, np.float32)

        nearest_tool_rel = self._nearest_spear_rel(agent_pos, agent.is_holding_tool)

        obs = np.concatenate([
            angles, vels, quat, lin_vel, ang_vel,
            vision,
            mammoth_rel,
            nearest_tool_rel,
            [agent.hunger / HUNGER_MAX,
             agent.stamina / STAMINA_MAX,
             float(agent.is_holding_tool)],
        ]).astype(np.float32)
        return obs

    def _cast_vision_rays(
        self, agent: Humanoid, mammoth_ids: set
    ) -> np.ndarray:
        """Cast VISION_RAYS forward-facing raycasts for one agent."""
        import math as _math
        head_pos = agent.get_head_position()
        head_ori = agent.get_head_orientation()
        mat      = p.getMatrixFromQuaternion(head_ori)
        forward  = np.array([mat[1], mat[4], mat[7]])
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0.0, 1.0, 0.0])
        else:
            forward /= norm
        right = np.array([-forward[1], forward[0], 0.0])

        half_fov = _math.radians(VISION_FOV_DEG / 2)
        froms, tos = [], []
        for i in range(VISION_RAYS):
            t     = i / max(VISION_RAYS - 1, 1)
            angle = -half_fov + t * 2 * half_fov
            ca, sa = _math.cos(angle), _math.sin(angle)
            d = (ca * forward + sa * right) * VISION_RANGE
            froms.append([head_pos[0], head_pos[1], head_pos[2]])
            tos.append([head_pos[0]+d[0], head_pos[1]+d[1], head_pos[2]+d[2]])

        results = p.rayTestBatch(
            froms, tos, physicsClientId=self._physics_client
        )
        out = np.zeros(VISION_RAYS * 2, dtype=np.float32)
        for i, res in enumerate(results):
            hid  = res[0]
            frac = float(res[2])
            if hid < 0:
                ht = HIT_NOTHING
            elif hid == self._terrain_id:
                ht = HIT_TERRAIN
            elif hid in mammoth_ids:
                ht = HIT_MAMMOTH
            elif hid == agent.body_id:
                ht = HIT_NOTHING
            else:
                ht = HIT_AGENT
            out[i*2]   = frac
            out[i*2+1] = ht
        return out

    def _nearest_spear_rel(
        self, agent_pos: np.ndarray, holding: bool
    ) -> np.ndarray:
        """Relative position to nearest un-held spear."""
        if holding or not self._spear_ids:
            return np.zeros(3, dtype=np.float32)
        best_d   = float('inf')
        best_pos = None
        for sid in self._spear_ids:
            try:
                spos, _ = p.getBasePositionAndOrientation(
                    sid, physicsClientId=self._physics_client
                )
            except Exception:
                continue
            d = float(np.linalg.norm(np.array(spos) - agent_pos))
            if d < best_d:
                best_d   = d
                best_pos = np.array(spos, dtype=np.float32)
        if best_pos is None:
            return np.zeros(3, dtype=np.float32)
        return best_pos - agent_pos

    def _predict_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        """Return (4, ACT_DIM) actions – policy or random."""
        if self.policy is None:
            return np.random.uniform(-1, 1, (AGENT_COUNT, ACT_DIM)).astype(np.float32)
        actions, _ = self.policy.predict(obs_batch, deterministic=True)
        return np.array(actions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Discrete action handlers
    # ------------------------------------------------------------------

    def _handle_discrete(
        self, agent: Humanoid, agent_idx: int, discrete: int
    ) -> None:
        """Execute the discrete action for one agent."""
        pos = agent.get_position()
        if discrete == 1 and not agent.is_holding_tool:
            self._try_grab(agent, pos, agent_idx)
        elif discrete == 2:
            self._try_attack(agent, pos)
        elif discrete == 3:
            self._try_eat(agent, pos)

    def _try_grab(
        self, agent: Humanoid, pos: np.ndarray, agent_idx: int
    ) -> None:
        for j, sid in enumerate(list(self._spear_ids)):
            try:
                spos, _ = p.getBasePositionAndOrientation(
                    sid, physicsClientId=self._physics_client
                )
            except Exception:
                continue
            if np.linalg.norm(np.array(spos) - pos) <= SPEAR_PICKUP_RADIUS:
                p.removeBody(sid, physicsClientId=self._physics_client)
                self._spear_ids.remove(sid)
                # Hide visual
                idx = j  # approximate – just hide first visible node
                if idx < len(self._spear_nodes) and self._spear_nodes[idx]:
                    self._spear_nodes[idx].hide()
                agent.grab_tool()
                return

    def _try_attack(self, agent: Humanoid, pos: np.ndarray) -> None:
        dmg = ARMED_DAMAGE if agent.is_holding_tool else UNARMED_DAMAGE
        for m in self.mammoth_mgr.mammoths:
            if m.is_dead:
                continue
            if np.linalg.norm(m.get_position() - pos) <= ATTACK_RADIUS:
                m.take_damage(dmg)
                return

    def _try_eat(self, agent: Humanoid, pos: np.ndarray) -> None:
        carcass, dist = self.mammoth_mgr.get_nearest_carcass(pos)
        if carcass is not None and dist <= EAT_RADIUS:
            agent.eat()

    # ------------------------------------------------------------------
    # Spear visual sync
    # ------------------------------------------------------------------

    def _sync_spear_visuals(self) -> None:
        """Sync spear Panda3D nodes to PyBullet positions."""
        for j, sid in enumerate(self._spear_ids):
            if j >= len(self._spear_nodes):
                break
            try:
                pos, ori = p.getBasePositionAndOrientation(
                    sid, physicsClientId=self._physics_client
                )
            except Exception:
                continue
            _set_np_transform(self._spear_nodes[j], pos, ori)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera(self) -> None:
        """Lerp the camera toward the selected agent."""
        if self.controls.free_cam:
            return

        idx = self.selected_agent
        if idx >= len(self.agents):
            return
        agent     = self.agents[idx]
        agent_pos = agent.get_position()

        _, ori = p.getBasePositionAndOrientation(
            agent.body_id, physicsClientId=self._physics_client
        )
        mat = p.getMatrixFromQuaternion(ori)
        fwd = np.array([mat[1], mat[4], mat[7]])
        fwd[2] = 0.0
        fn = np.linalg.norm(fwd)
        if fn > 1e-6:
            fwd /= fn
        else:
            fwd = np.array([0.0, 1.0, 0.0])

        desired_pos = Vec3(
            float(agent_pos[0] - fwd[0] * CAMERA_DISTANCE),
            float(agent_pos[1] - fwd[1] * CAMERA_DISTANCE),
            float(agent_pos[2] + CAMERA_HEIGHT),
        )
        desired_look = Vec3(
            float(agent_pos[0]),
            float(agent_pos[1]),
            float(agent_pos[2] + 0.5),
        )

        self._cam_pos  = self._cam_pos  + (desired_pos  - self._cam_pos)  * CAMERA_LERP
        self._cam_look = self._cam_look + (desired_look - self._cam_look) * CAMERA_LERP

        self.camera.setPos(self._cam_pos)
        self.camera.lookAt(self._cam_look)

    # ------------------------------------------------------------------
    # HUD update
    # ------------------------------------------------------------------

    def _update_hud(self) -> None:
        """Collect per-agent stats and pass to HUD.update()."""
        stats = []
        for agent in self.agents:
            if agent.is_dead:
                label = 'DEAD'
            elif agent.is_holding_tool:
                label = 'HUNTING'
            else:
                label = 'IDLE'
            stats.append({
                'hunger':          agent.hunger,
                'stamina':         agent.stamina,
                'is_holding_tool': agent.is_holding_tool,
                'is_dead':         agent.is_dead,
                'status_label':    label,
            })
        self.hud.update(
            agent_stats   = stats,
            selected_agent= self.selected_agent,
            sim_time      = self._sim_time,
            mammoths_alive= self.mammoth_mgr.count_alive(),
            agents_alive  = sum(1 for a in self.agents if not a.is_dead),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_world(self) -> None:
        """Full episode reset: respawn all agents, mammoths, and spears."""
        # Remove old bodies
        self.mammoth_mgr.clear()
        for sid in self._spear_ids:
            try:
                p.removeBody(sid, physicsClientId=self._physics_client)
            except Exception:
                pass
        self._spear_ids.clear()
        for n in self._spear_nodes:
            try:
                n.removeNode()
            except Exception:
                pass
        self._spear_nodes.clear()

        # Reset agents
        for i, agent in enumerate(self.agents):
            x = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
            y = random.uniform(-AGENT_SPAWN_RADIUS, AGENT_SPAWN_RADIUS)
            z = (self._get_terrain_height(x, y)
                 + AGENT_SPAWN_Z_OFFSET + PELVIS_HEIGHT_ABOVE_FEET)
            agent.reset([x, y, z])

        # Respawn mammoths and spears
        self._spawn_initial_mammoths()
        self._spawn_spears()
        self._stabilise()
        self._sim_time = 0.0
        self._step_num = 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_terrain_height(self, x: float, y: float) -> float:
        """Raycast downward to find terrain surface height at (x, y)."""
        result = p.rayTest(
            [x, y, TERRAIN_AMPLITUDE + 5.0],
            [x, y, -10.0],
            physicsClientId=self._physics_client,
        )
        if result and result[0][0] >= 0:
            return float(result[0][3][2])
        return 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and launch the visual simulation."""
    parser = argparse.ArgumentParser(
        description='Primal Survival Simulation – Visual Play Mode',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to .zip policy checkpoint file.',
    )
    parser.add_argument(
        '--random', action='store_true',
        help='Use random actions (ignore any checkpoint).',
    )
    args = parser.parse_args()

    policy = None
    if not args.random:
        from agent_brain import load_policy_for_play
        policy = load_policy_for_play(args.checkpoint)

    app = PrimalSimApp(policy=policy)
    app.run()


if __name__ == '__main__':
    main()
