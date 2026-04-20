"""All tunable constants for the Primal Survival Simulation.

No imports from other project files. Every magic number lives here.
"""

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
WORLD_SIZE = 1000          # metres, sandbox is WORLD_SIZE × WORLD_SIZE
WALL_HEIGHT = 50           # metres, invisible boundary wall height
TERRAIN_SIZE = 256         # heightmap grid resolution (256 × 256)
TERRAIN_AMPLITUDE = 8.0    # max hill height in metres
TERRAIN_NOISE_SCALE = 0.018  # Perlin frequency (lower = broader hills)
TERRAIN_SCALE_XY = WORLD_SIZE / TERRAIN_SIZE   # metres per heightmap cell ≈ 3.9 m

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
AGENT_COUNT = 4
AGENT_MASS = 70.0          # kg, total body mass
AGENT_SPAWN_RADIUS = 50.0  # metres from origin, random placement at start
AGENT_COLORS = [           # RGBA solid colours per agent
    (0.95, 0.15, 0.15, 1.0),   # Agent 0 – bright red
    (0.15, 0.40, 1.00, 1.0),   # Agent 1 – bright blue
    (1.00, 0.90, 0.10, 1.0),   # Agent 2 – bright yellow
    (0.10, 0.90, 0.25, 1.0),   # Agent 3 – bright green
]

# Humanoid standing height above spawn surface
AGENT_SPAWN_Z_OFFSET = 5   # metres above terrain surface

# ---------------------------------------------------------------------------
# Agent stats
# ---------------------------------------------------------------------------
HUNGER_START = 10.0           # 0–100 scale
HUNGER_MAX = 100.0
HUNGER_DRAIN_PER_STEP = 0.001 # per physics step (very slow)
HUNGER_RESTORE_ON_EAT = 100.0 # restored to this value when eating
STARVATION_THRESHOLD = 20.0   # below this hunger → starvation penalty fires
STAMINA_MAX = 100.0
STAMINA_DRAIN_HIGH_TORQUE = 0.05   # per step when joint torques are large
STAMINA_RECOVER_RATE = 0.1         # per step when idle
HIGH_TORQUE_THRESHOLD = 0.7        # fraction of max torque = "sprinting"

# Upright detection: torso z-component of up-vector must exceed this
UPRIGHT_THRESHOLD = 0.5   # cos of ~60 degrees

# ---------------------------------------------------------------------------
# Spears / tools
# ---------------------------------------------------------------------------
SPEAR_COUNT = 8
SPEAR_SPAWN_RADIUS = 100.0   # max distance from origin
SPEAR_PICKUP_RADIUS = 1.5    # metres, grab distance
SPEAR_MASS = 2.0             # kg
SPEAR_HALF_LENGTH = 1.0      # metres (total length = 2 m)
SPEAR_RADIUS = 0.04          # cylinder radius

# ---------------------------------------------------------------------------
# Mammoth
# ---------------------------------------------------------------------------
MAMMOTH_MASS = 5000.0        # kg
MAMMOTH_SPEED = 3.0          # m/s walk speed
MAMMOTH_HP = 100.0
MAMMOTH_LINEAR_DAMPING = 0.9
MAMMOTH_ANGULAR_DAMPING = 0.99
MAMMOTH_AVOIDANCE_DIST = 60.0        # metres, flee radius from agents
MAMMOTH_AVOIDANCE_RECALC_STEPS = 10  # how often to recalculate steering
MAMMOTH_WAYPOINT_INTERVAL = 10.0     # seconds between random waypoints
MAMMOTH_CARCASS_LIFETIME = 120.0     # seconds the carcass persists
MAMMOTH_BODY_HALF_EXTENTS = (2.0, 3.0, 1.25)   # x,y,z half-extents (4×6×2.5 m)
MAMMOTH_LEG_HALF_EXTENTS  = (0.3, 0.3, 0.8)
MAMMOTH_TUSK_HALF_EXTENTS = (0.15, 0.8, 0.15)
MAMMOTH_SPAWN_RADIUS = 400.0         # random spawn distance from origin
MAMMOTH_PD_KP = 8000.0    # proportional gain for velocity PD controller
MAMMOTH_PD_KD = 2000.0    # derivative gain
MAMMOTH_MAX_FORCE = 80000.0  # N, clamp on PD output

# Default mammoth count spawned at episode start
MAMMOTH_INITIAL_COUNT = 2

# ---------------------------------------------------------------------------
# Combat / eating
# ---------------------------------------------------------------------------
ARMED_DAMAGE = 25.0
UNARMED_DAMAGE = 5.0
ATTACK_RADIUS = 2.0   # metres
EAT_RADIUS = 10.0     # metres from carcass to trigger eat

# ---------------------------------------------------------------------------
# Perception (vision system)
# ---------------------------------------------------------------------------
VISION_FOV_DEG = 120.0   # total horizontal FOV in degrees
VISION_RANGE = 200.0     # metres
VISION_RAYS = 16         # rays across FOV
# Vision hit-type encoding (0–1 float)
HIT_NOTHING  = 0.0
HIT_TERRAIN  = 0.33
HIT_MAMMOTH  = 0.66
HIT_AGENT    = 1.0

# ---------------------------------------------------------------------------
# Observation / action dimensions
# ---------------------------------------------------------------------------
OBS_DIM = 75   # total floats in observation vector
ACT_DIM = 13   # 12 joint torques + 1 discrete-as-continuous

# Discrete action buckets encoded in action[12]  ∈ [-1, 1]:
#   bucket 0 = [-1.0, -0.5)  → nothing
#   bucket 1 = [-0.5,  0.0)  → grab_nearby_tool
#   bucket 2 = [ 0.0,  0.5)  → attack_nearby_mammoth
#   bucket 3 = [ 0.5,  1.0]  → eat_nearby_carcass
DISCRETE_BUCKET_EDGES = [-1.0, -0.5, 0.0, 0.5, 1.0]

# ---------------------------------------------------------------------------
# Humanoid skeleton (12 revolute joints, 12 links)
# ---------------------------------------------------------------------------
# Link index constants
LINK_UPPER_TORSO = 0
LINK_HEAD        = 1
LINK_L_UPPER_ARM = 2
LINK_L_LOWER_ARM = 3
LINK_R_UPPER_ARM = 4
LINK_R_LOWER_ARM = 5
LINK_L_UPPER_LEG = 6
LINK_L_LOWER_LEG = 7
LINK_L_FOOT      = 8
LINK_R_UPPER_LEG = 9
LINK_R_LOWER_LEG = 10
LINK_R_FOOT      = 11
NUM_JOINTS       = 12

# Per-link masses (kg) – sum + BASE_MASS ≈ 70 kg
LINK_MASSES = [
    18.0,  # upper_torso
     5.0,  # head
     2.0,  # l_upper_arm
     1.5,  # l_lower_arm
     2.0,  # r_upper_arm
     1.5,  # r_lower_arm
     8.0,  # l_upper_leg
     4.0,  # l_lower_leg
     1.0,  # l_foot
     8.0,  # r_upper_leg
     4.0,  # r_lower_leg
     1.0,  # r_foot
]
BASE_MASS = 10.0  # pelvis

# Joint parent indices for createMultiBody
# 0 = base (pelvis), k+1 = link at array index k
LINK_PARENT_INDICES = [0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11]

# Maximum joint torque (N·m) applied by RL policy
MAX_JOINT_TORQUE = 200.0

# Joint axis (all revolute around X-axis = sagittal/frontal plane)
JOINT_AXIS = [1, 0, 0]

# Joint angle limits (radians) – (lower, upper)
JOINT_LIMITS = [
    (-0.5, 0.5),   # 0 waist
    (-0.5, 0.5),   # 1 neck
    (-1.5, 0.5),   # 2 l_shoulder
    (-2.5, 0.0),   # 3 l_elbow
    (-1.5, 0.5),   # 4 r_shoulder
    ( 0.0, 2.5),   # 5 r_elbow
    (-0.5, 1.5),   # 6 l_hip
    ( 0.0, 2.5),   # 7 l_knee
    (-0.5, 0.5),   # 8 l_ankle
    (-0.5, 1.5),   # 9 r_hip
    ( 0.0, 2.5),   # 10 r_knee
    (-0.5, 0.5),   # 11 r_ankle
]

# ---------------------------------------------------------------------------
# Camera (play mode)
# ---------------------------------------------------------------------------
CAMERA_DISTANCE = 8.0   # metres behind agent
CAMERA_HEIGHT   = 5.0   # metres above agent
CAMERA_LERP     = 0.08  # position interpolation factor per frame
FREE_CAM_SPEED  = 12.0  # WASD movement speed m/s

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
PHYSICS_TIMESTEP  = 1.0 / 60.0   # seconds
PHYSICS_SUBSTEPS  = 4
SETTLE_STEPS      = 60   # zero-torque steps at episode start

# ---------------------------------------------------------------------------
# RL / training
# ---------------------------------------------------------------------------
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS       = 2048
PPO_BATCH_SIZE    = 256
PPO_N_EPOCHS      = 10
PPO_GAMMA         = 0.99
N_ENVS            = 4
MAX_EPISODE_STEPS = 4096   # truncation limit per episode

CHECKPOINT_DIR  = './checkpoints'
LOG_DIR         = './logs'
CHECKPOINT_FREQ = 50_000   # total env steps between saves

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
REWARD_UPRIGHT              =  0.10
REWARD_TOWARD_MAMMOTH       =  0.20   # per metre closed
PENALTY_AWAY_MAMMOTH        = -0.05   # per step moving away when mammoth visible
REWARD_FIRST_TOOL           =  0.50   # one-time curiosity bonus
REWARD_HIT_MAMMOTH          =  2.00
REWARD_KILL_MAMMOTH         = 10.00
REWARD_EAT_CARCASS          =  5.00
PENALTY_STARVATION          = -1.00   # per step when hunger < threshold
PENALTY_WASTED_SPRINT       = -0.01   # per step sprinting with no mammoth visible
PENALTY_DEATH               = -100.00
