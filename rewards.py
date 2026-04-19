"""Pure reward-shaping functions for the Primal Survival Simulation.

All functions are stateless and operate on scalar / vector inputs.
No PyBullet, Panda3D, or other project imports — only config constants.
"""

import math
import numpy as np

from config import (
    REWARD_UPRIGHT,
    REWARD_TOWARD_MAMMOTH,
    PENALTY_AWAY_MAMMOTH,
    REWARD_FIRST_TOOL,
    REWARD_HIT_MAMMOTH,
    REWARD_KILL_MAMMOTH,
    REWARD_EAT_CARCASS,
    PENALTY_STARVATION,
    PENALTY_WASTED_SPRINT,
    PENALTY_DEATH,
    STARVATION_THRESHOLD,
    UPRIGHT_THRESHOLD,
)


def reward_upright(up_dot_z: float) -> float:
    """Reward for staying upright.

    Args:
        up_dot_z: Dot product of agent's local up-vector with world Z (range -1..1).
                  1.0 means perfectly upright, -1.0 means inverted.
    """
    if up_dot_z >= UPRIGHT_THRESHOLD:
        return REWARD_UPRIGHT * up_dot_z
    return 0.0


def reward_locomotion(
    current_pos: np.ndarray,
    prev_pos: np.ndarray,
    mammoth_pos: np.ndarray | None,
) -> float:
    """Reward / penalise movement relative to the nearest visible mammoth.

    Returns 0 when no mammoth is visible.
    """
    if mammoth_pos is None:
        return 0.0

    prev_dist = np.linalg.norm(mammoth_pos - prev_pos)
    curr_dist = np.linalg.norm(mammoth_pos - current_pos)
    delta = prev_dist - curr_dist  # positive means closing

    if delta > 0:
        return REWARD_TOWARD_MAMMOTH * delta
    return PENALTY_AWAY_MAMMOTH


def reward_first_tool(just_picked_up: bool) -> float:
    """One-time curiosity bonus when the agent first picks up a tool.

    The caller is responsible for calling this only on the single step
    where the tool is grabbed for the first time.
    """
    return REWARD_FIRST_TOOL if just_picked_up else 0.0


def reward_hit_mammoth(did_hit: bool) -> float:
    """Positive reward for landing a hit on a mammoth."""
    return REWARD_HIT_MAMMOTH if did_hit else 0.0


def reward_kill_mammoth(just_killed: bool) -> float:
    """Large reward for killing a mammoth."""
    return REWARD_KILL_MAMMOTH if just_killed else 0.0


def reward_eat_carcass(did_eat: bool) -> float:
    """Reward for eating a mammoth carcass."""
    return REWARD_EAT_CARCASS if did_eat else 0.0


def penalty_starvation(hunger: float) -> float:
    """Per-step penalty when the agent is starving."""
    return PENALTY_STARVATION if hunger < STARVATION_THRESHOLD else 0.0


def penalty_wasted_sprint(
    is_high_torque: bool, mammoth_visible: bool
) -> float:
    """Penalty for burning stamina with no mammoth in sight."""
    if is_high_torque and not mammoth_visible:
        return PENALTY_WASTED_SPRINT
    return 0.0


def penalty_death(is_dead: bool) -> float:
    """Terminal penalty on agent death."""
    return PENALTY_DEATH if is_dead else 0.0


def compute_total_reward(
    up_dot_z: float,
    current_pos: np.ndarray,
    prev_pos: np.ndarray,
    mammoth_pos: np.ndarray | None,
    just_picked_up_first_tool: bool,
    did_hit: bool,
    just_killed: bool,
    did_eat: bool,
    hunger: float,
    is_high_torque: bool,
    mammoth_visible: bool,
    is_dead: bool,
) -> float:
    """Aggregate all reward components into a single scalar."""
    r = 0.0
    r += reward_upright(up_dot_z)
    r += reward_locomotion(current_pos, prev_pos, mammoth_pos)
    r += reward_first_tool(just_picked_up_first_tool)
    r += reward_hit_mammoth(did_hit)
    r += reward_kill_mammoth(just_killed)
    r += reward_eat_carcass(did_eat)
    r += penalty_starvation(hunger)
    r += penalty_wasted_sprint(is_high_torque, mammoth_visible)
    r += penalty_death(is_dead)
    return float(r)
