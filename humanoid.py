"""Humanoid agent – PyBullet physics body and optional Panda3D visual sync.

One Humanoid instance represents a single agent.  It owns its PyBullet body
and all Panda3D NodePaths (when running in play mode).
"""

import math
import numpy as np
import pybullet as p

from config import (
    BASE_MASS, LINK_MASSES, LINK_PARENT_INDICES,
    NUM_JOINTS, JOINT_AXIS, JOINT_LIMITS, MAX_JOINT_TORQUE,
    LINK_UPPER_TORSO, LINK_HEAD,
    LINK_L_UPPER_ARM, LINK_L_LOWER_ARM,
    LINK_R_UPPER_ARM, LINK_R_LOWER_ARM,
    LINK_L_UPPER_LEG, LINK_L_LOWER_LEG, LINK_L_FOOT,
    LINK_R_UPPER_LEG, LINK_R_LOWER_LEG, LINK_R_FOOT,
    HUNGER_START, HUNGER_MAX, HUNGER_DRAIN_PER_STEP,
    STAMINA_MAX, STAMINA_DRAIN_HIGH_TORQUE, STAMINA_RECOVER_RATE,
    HIGH_TORQUE_THRESHOLD, DISCRETE_BUCKET_EDGES,
    VISION_FOV_DEG, VISION_RANGE, VISION_RAYS,
    HIT_NOTHING, HIT_TERRAIN, HIT_MAMMOTH, HIT_AGENT,
    UPRIGHT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Collision shape half-extents / radii (metres)
# ---------------------------------------------------------------------------
_BASE_CAPSULE_R = 0.13
_BASE_CAPSULE_H = 0.16

_TORSO_CAPSULE_R = 0.15
_TORSO_CAPSULE_H = 0.28

_HEAD_SPHERE_R = 0.12

_UPPER_ARM_R = 0.06
_UPPER_ARM_H = 0.26

_LOWER_ARM_R = 0.05
_LOWER_ARM_H = 0.22

_UPPER_LEG_R = 0.09
_UPPER_LEG_H = 0.36

_LOWER_LEG_R = 0.07
_LOWER_LEG_H = 0.32

_FOOT_BOX = [0.12, 0.24, 0.05]   # half-extents x, y, z


# ---------------------------------------------------------------------------
# Joint positions in parent frame (where the joint pivot lives)
# ---------------------------------------------------------------------------
#   These define the kinematic chain offsets.
#   The pelvis (base) COM is the origin; everything is relative.
_LINK_POSITIONS = [
    # 0: upper_torso – waist joint 18 cm above pelvis COM
    ( 0.00,  0.00,  0.18),
    # 1: head – neck joint 28 cm above upper_torso COM
    ( 0.00,  0.00,  0.28),
    # 2: l_upper_arm – left shoulder, 22 cm left, 22 cm up from torso COM
    (-0.22,  0.00,  0.22),
    # 3: l_lower_arm – elbow 30 cm below shoulder joint
    ( 0.00,  0.00, -0.30),
    # 4: r_upper_arm – right shoulder
    ( 0.22,  0.00,  0.22),
    # 5: r_lower_arm – elbow 30 cm below
    ( 0.00,  0.00, -0.30),
    # 6: l_upper_leg – left hip, 11 cm left, 13 cm below pelvis COM
    (-0.11,  0.00, -0.13),
    # 7: l_lower_leg – knee 38 cm below hip
    ( 0.00,  0.00, -0.38),
    # 8: l_foot – ankle 36 cm below knee
    ( 0.00,  0.00, -0.36),
    # 9: r_upper_leg – right hip
    ( 0.11,  0.00, -0.13),
    # 10: r_lower_leg – knee 38 cm below hip
    ( 0.00,  0.00, -0.38),
    # 11: r_foot – ankle 36 cm below knee
    ( 0.00,  0.00, -0.36),
]

# Inertial frame offsets (COM relative to joint position) – all at joint
_LINK_INERTIAL = [(0.0, 0.0, 0.0)] * NUM_JOINTS
_LINK_INERTIAL_ORIS = [(0.0, 0.0, 0.0, 1.0)] * NUM_JOINTS

_IDENTITY_ORI = (0.0, 0.0, 0.0, 1.0)

# Approximate COM height of the body above the pelvis spawn point
# (used by environment to place the agent at the right initial Z)
PELVIS_HEIGHT_ABOVE_FEET = 0.93  # metres (hip to foot when standing)


# ---------------------------------------------------------------------------
# Panda3D visual dimensions (half-extents for box nodes)
# ---------------------------------------------------------------------------
_VIS_TORSO   = (0.14, 0.11, 0.24)
_VIS_HEAD    = (0.10, 0.10, 0.10)
_VIS_UPPER_ARM  = (0.06, 0.06, 0.14)
_VIS_LOWER_ARM  = (0.05, 0.05, 0.12)
_VIS_UPPER_LEG  = (0.09, 0.09, 0.20)
_VIS_LOWER_LEG  = (0.07, 0.07, 0.18)
_VIS_PELVIS  = (0.12, 0.10, 0.10)
_VIS_FOOT    = (0.12, 0.24, 0.05)


class Humanoid:
    """A single humanoid agent with physics body and optional Panda3D visuals."""

    def __init__(
        self,
        physics_client: int,
        start_pos: list,
        color_rgba: tuple,
        agent_id: int = 0,
        render_parent=None,   # Panda3D NodePath or None
    ):
        """Create the PyBullet body and, if render_parent is given, Panda3D nodes.

        Args:
            physics_client: PyBullet client ID.
            start_pos:       [x, y, z] spawn position for the pelvis.
            color_rgba:      (r, g, b, a) solid colour for visuals.
            agent_id:        Index used for collision filtering (0-3).
            render_parent:   Panda3D NodePath to attach visuals to, or None.
        """
        self.client = physics_client
        self.color = color_rgba
        self.agent_id = agent_id

        # Stats
        self.hunger: float  = HUNGER_START
        self.stamina: float = STAMINA_MAX
        self.is_holding_tool: bool = False
        self.is_dead: bool = False
        self._first_tool_grabbed: bool = False

        self._prev_pos: np.ndarray = np.array(start_pos, dtype=np.float32)

        # Build physics body
        self.body_id: int = self._build_body(start_pos)

        # Disable default velocity motors so torque control works cleanly
        for j in range(NUM_JOINTS):
            p.setJointMotorControl2(
                self.body_id, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client,
            )

        # Panda3D visuals
        self._vis_nodes: dict = {}  # name -> NodePath
        self._render_parent = render_parent
        if render_parent is not None:
            self._build_visuals(render_parent)

    # ------------------------------------------------------------------
    # Physics body construction
    # ------------------------------------------------------------------

    def _build_body(self, start_pos: list) -> int:
        """Construct the 12-link articulated body in PyBullet."""
        # Collision shapes
        base_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_BASE_CAPSULE_R,
            height=_BASE_CAPSULE_H,
            physicsClientId=self.client,
        )
        torso_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_TORSO_CAPSULE_R,
            height=_TORSO_CAPSULE_H,
            physicsClientId=self.client,
        )
        head_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=_HEAD_SPHERE_R,
            physicsClientId=self.client,
        )
        u_arm_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_UPPER_ARM_R,
            height=_UPPER_ARM_H,
            physicsClientId=self.client,
        )
        l_arm_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_LOWER_ARM_R,
            height=_LOWER_ARM_H,
            physicsClientId=self.client,
        )
        u_leg_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_UPPER_LEG_R,
            height=_UPPER_LEG_H,
            physicsClientId=self.client,
        )
        l_leg_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=_LOWER_LEG_R,
            height=_LOWER_LEG_H,
            physicsClientId=self.client,
        )
        foot_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=_FOOT_BOX,
            physicsClientId=self.client,
        )

        link_shapes = [
            torso_shape,    # 0 upper_torso
            head_shape,     # 1 head
            u_arm_shape,    # 2 l_upper_arm
            l_arm_shape,    # 3 l_lower_arm
            u_arm_shape,    # 4 r_upper_arm
            l_arm_shape,    # 5 r_lower_arm
            u_leg_shape,    # 6 l_upper_leg
            l_leg_shape,    # 7 l_lower_leg
            foot_shape,     # 8 l_foot
            u_leg_shape,    # 9 r_upper_leg
            l_leg_shape,    # 10 r_lower_leg
            foot_shape,     # 11 r_foot
        ]

        link_visual_shapes = [-1] * NUM_JOINTS
        link_orientations = [_IDENTITY_ORI] * NUM_JOINTS
        link_joint_axes = [JOINT_AXIS] * NUM_JOINTS

        body_id = p.createMultiBody(
            baseMass=BASE_MASS,
            baseCollisionShapeIndex=base_shape,
            baseVisualShapeIndex=-1,
            basePosition=start_pos,
            baseOrientation=_IDENTITY_ORI,
            linkMasses=LINK_MASSES,
            linkCollisionShapeIndices=link_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=_LINK_POSITIONS,
            linkOrientations=link_orientations,
            linkInertialFramePositions=_LINK_INERTIAL,
            linkInertialFrameOrientations=_LINK_INERTIAL_ORIS,
            linkParentIndices=LINK_PARENT_INDICES,
            linkJointTypes=[p.JOINT_REVOLUTE] * NUM_JOINTS,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.client,
        )

        # Apply joint angle limits
        for j, (lo, hi) in enumerate(JOINT_LIMITS):
            p.changeDynamics(
                body_id, j,
                jointLowerLimit=lo,
                jointUpperLimit=hi,
                physicsClientId=self.client,
            )

        # Friction on feet
        for link_idx in (LINK_L_FOOT, LINK_R_FOOT, -1):
            p.changeDynamics(
                body_id, link_idx,
                lateralFriction=1.0,
                spinningFriction=0.1,
                physicsClientId=self.client,
            )

        return body_id

    # ------------------------------------------------------------------
    # Panda3D visual construction
    # ------------------------------------------------------------------

    def _build_visuals(self, parent_np) -> None:
        """Create coloured box/sphere Panda3D nodes for each body segment."""
        from play_utils import make_box_np, make_sphere_np  # lazy import

        r, g, b, a = self.color

        def _attach(name: str, np_node):
            np_node.reparentTo(parent_np)
            self._vis_nodes[name] = np_node

        _attach('pelvis',     make_box_np(*_VIS_PELVIS,    (r, g, b, a)))
        _attach('torso',      make_box_np(*_VIS_TORSO,     (r, g, b, a)))
        _attach('head',       make_sphere_np(_HEAD_SPHERE_R, (r, g, b, a)))
        _attach('l_u_arm',    make_box_np(*_VIS_UPPER_ARM, (r, g, b, a)))
        _attach('l_l_arm',    make_box_np(*_VIS_LOWER_ARM, (r, g, b, a)))
        _attach('r_u_arm',    make_box_np(*_VIS_UPPER_ARM, (r, g, b, a)))
        _attach('r_l_arm',    make_box_np(*_VIS_LOWER_ARM, (r, g, b, a)))
        _attach('l_u_leg',    make_box_np(*_VIS_UPPER_LEG, (r, g, b, a)))
        _attach('l_l_leg',    make_box_np(*_VIS_LOWER_LEG, (r, g, b, a)))
        _attach('l_foot',     make_box_np(*_VIS_FOOT,      (r, g, b, a)))
        _attach('r_u_leg',    make_box_np(*_VIS_UPPER_LEG, (r, g, b, a)))
        _attach('r_l_leg',    make_box_np(*_VIS_LOWER_LEG, (r, g, b, a)))
        _attach('r_foot',     make_box_np(*_VIS_FOOT,      (r, g, b, a)))

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def apply_action(self, action: np.ndarray) -> int:
        """Apply joint torques and decode discrete action.

        Args:
            action: shape (13,). action[:12] are normalised torques [-1, 1].
                    action[12] is bucketed to discrete action index 0-3.

        Returns:
            discrete_action (int): 0=nothing, 1=grab, 2=attack, 3=eat.
        """
        if self.is_dead:
            return 0

        torques = np.clip(action[:12], -1.0, 1.0) * MAX_JOINT_TORQUE
        for j in range(NUM_JOINTS):
            p.setJointMotorControl2(
                self.body_id, j,
                controlMode=p.TORQUE_CONTROL,
                force=float(torques[j]),
                physicsClientId=self.client,
            )

        # Track stamina
        mean_effort = float(np.mean(np.abs(action[:12])))
        if mean_effort > HIGH_TORQUE_THRESHOLD:
            self.stamina = max(0.0, self.stamina - STAMINA_DRAIN_HIGH_TORQUE)
            self._is_high_torque = True
        else:
            self.stamina = min(STAMINA_MAX, self.stamina + STAMINA_RECOVER_RATE)
            self._is_high_torque = False

        # Decode discrete action
        val = float(action[12])
        edges = DISCRETE_BUCKET_EDGES
        if val < edges[1]:
            return 0
        elif val < edges[2]:
            return 1
        elif val < edges[3]:
            return 2
        return 3

    def drain_hunger(self) -> None:
        """Called each physics step to reduce hunger."""
        if self.is_dead:
            return
        self.hunger = max(0.0, self.hunger - HUNGER_DRAIN_PER_STEP)
        if self.hunger <= 0.0:
            self.is_dead = True

    def get_position(self) -> np.ndarray:
        """World position of the pelvis base."""
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        return np.array(pos, dtype=np.float32)

    def update_prev_position(self) -> None:
        """Cache current position before stepping (for locomotion reward)."""
        self._prev_pos = self.get_position()

    def get_prev_position(self) -> np.ndarray:
        """Position cached at the start of the current step."""
        return self._prev_pos.copy()

    def get_up_dot_z(self) -> float:
        """Return dot product of agent local up-axis with world Z.

        Used by the upright reward. 1.0 = perfectly upright.
        """
        _, ori = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        mat = p.getMatrixFromQuaternion(ori)
        # Local Z column of the rotation matrix (indices 2, 5, 8)
        local_z = np.array([mat[2], mat[5], mat[8]])
        return float(local_z[2])   # dot with world Z = z component

    def get_head_position(self) -> np.ndarray:
        """World position of the head link (for vision ray origin)."""
        state = p.getLinkState(
            self.body_id, LINK_HEAD,
            computeForwardKinematics=1,
            physicsClientId=self.client,
        )
        return np.array(state[4], dtype=np.float32)

    def get_head_orientation(self) -> np.ndarray:
        """World orientation (x,y,z,w) of the head link."""
        state = p.getLinkState(
            self.body_id, LINK_HEAD,
            computeForwardKinematics=1,
            physicsClientId=self.client,
        )
        return np.array(state[5], dtype=np.float32)

    def get_joint_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (angles, velocities) arrays of shape (12,)."""
        angles = np.zeros(NUM_JOINTS, dtype=np.float32)
        vels   = np.zeros(NUM_JOINTS, dtype=np.float32)
        for j in range(NUM_JOINTS):
            state = p.getJointState(
                self.body_id, j, physicsClientId=self.client
            )
            angles[j] = state[0]
            vels[j]   = state[1]
        return angles, vels

    def get_base_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (quaternion, lin_vel, ang_vel) as numpy arrays."""
        pos, ori = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.body_id, physicsClientId=self.client
        )
        return (
            np.array(ori,     dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel, dtype=np.float32),
        )

    def is_high_torque_step(self) -> bool:
        """True if last apply_action() was a high-torque (sprint) step."""
        return getattr(self, '_is_high_torque', False)

    def eat(self) -> None:
        """Restore hunger to maximum on eating a carcass."""
        self.hunger = HUNGER_MAX

    def grab_tool(self) -> bool:
        """Mark agent as holding a tool. Returns True if this is the first time."""
        was_first = not self._first_tool_grabbed
        self.is_holding_tool = True
        self._first_tool_grabbed = True
        return was_first

    def drop_tool(self) -> None:
        """Release the tool."""
        self.is_holding_tool = False

    # ------------------------------------------------------------------
    # Panda3D visual sync
    # ------------------------------------------------------------------

    def sync_visuals(self) -> None:
        """Update all Panda3D NodePath transforms to match PyBullet state.

        Must be called once per rendered frame in play mode.
        """
        if not self._vis_nodes:
            return

        from panda3d.core import LPoint3f, LQuaternionf

        # Base (pelvis)
        pos, ori = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        _set_np_transform(self._vis_nodes['pelvis'], pos, ori)

        # Links
        link_name_map = {
            LINK_UPPER_TORSO: 'torso',
            LINK_HEAD:        'head',
            LINK_L_UPPER_ARM: 'l_u_arm',
            LINK_L_LOWER_ARM: 'l_l_arm',
            LINK_R_UPPER_ARM: 'r_u_arm',
            LINK_R_LOWER_ARM: 'r_l_arm',
            LINK_L_UPPER_LEG: 'l_u_leg',
            LINK_L_LOWER_LEG: 'l_l_leg',
            LINK_L_FOOT:      'l_foot',
            LINK_R_UPPER_LEG: 'r_u_leg',
            LINK_R_LOWER_LEG: 'r_l_leg',
            LINK_R_FOOT:      'r_foot',
        }
        for link_idx, name in link_name_map.items():
            if name not in self._vis_nodes:
                continue
            state = p.getLinkState(
                self.body_id, link_idx,
                computeForwardKinematics=1,
                physicsClientId=self.client,
            )
            _set_np_transform(self._vis_nodes[name], state[4], state[5])

        # Hide body if dead
        is_hidden = self._vis_nodes['pelvis'].isHidden()
        if self.is_dead and not is_hidden:
            for np_node in self._vis_nodes.values():
                np_node.hide()
        elif not self.is_dead and is_hidden:
            for np_node in self._vis_nodes.values():
                np_node.show()

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------

    def reset(self, start_pos: list) -> None:
        """Respawn agent at start_pos with fresh stats."""
        self.hunger  = HUNGER_START
        self.stamina = STAMINA_MAX
        self.is_holding_tool = False
        self.is_dead = False
        self._first_tool_grabbed = False
        self._prev_pos = np.array(start_pos, dtype=np.float32)
        p.resetBasePositionAndOrientation(
            self.body_id, start_pos, _IDENTITY_ORI,
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.body_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client,
        )
        for j in range(NUM_JOINTS):
            p.resetJointState(
                self.body_id, j, 0.0, 0.0,
                physicsClientId=self.client,
            )

    def remove(self) -> None:
        """Remove PyBullet body and hide Panda3D nodes."""
        try:
            p.removeBody(self.body_id, physicsClientId=self.client)
        except Exception:
            pass
        for np_node in self._vis_nodes.values():
            try:
                np_node.removeNode()
            except Exception:
                pass
        self._vis_nodes.clear()


# ---------------------------------------------------------------------------
# Coordinate conversion helper (used by sync_visuals and mammoth.py)
# ---------------------------------------------------------------------------

def _set_np_transform(node_path, pos, ori_xyzw) -> None:
    """Set a Panda3D NodePath position + orientation from PyBullet state.

    Args:
        pos:       (x, y, z) world position.
        ori_xyzw:  (x, y, z, w) quaternion from PyBullet.
    """
    from panda3d.core import LPoint3f, LQuaternionf
    node_path.setPos(LPoint3f(float(pos[0]), float(pos[1]), float(pos[2])))
    # Panda3D Quat is (w, x, y, z)
    node_path.setQuat(
        LQuaternionf(
            float(ori_xyzw[3]),
            float(ori_xyzw[0]),
            float(ori_xyzw[1]),
            float(ori_xyzw[2]),
        )
    )
