"""RL policy setup – PPO factory, SubprocVecEnv construction, checkpoint I/O.

This module contains no training loops; those live in train.py.
"""

import os
import json
import shutil
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from config import (
    PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_GAMMA, N_ENVS,
    CHECKPOINT_DIR, LOG_DIR, CHECKPOINT_FREQ,
    AGENT_COUNT,
)
from environment import PrimalSurvivalEnv


def _get_device(device: str = 'auto') -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' depending on availability."""
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def make_env_factory(agent_index: int):
    """Return a no-arg callable that constructs a single PrimalSurvivalEnv.

    Each subprocess must call this factory to build its own env and its own
    PyBullet DIRECT client.
    """
    def _init() -> PrimalSurvivalEnv:
        env = PrimalSurvivalEnv(agent_index=agent_index)
        return env
    return _init


def build_vec_env(n_envs: int = N_ENVS) -> SubprocVecEnv:
    """Construct a SubprocVecEnv with *n_envs* parallel environments.

    Each worker gets a distinct agent_index so they use different colours and
    different random seeds inside the env.
    """
    factories = [make_env_factory(i % AGENT_COUNT) for i in range(n_envs)]
    return SubprocVecEnv(factories, start_method='spawn')


def build_ppo(vec_env: VecEnv, device: str = 'auto') -> PPO:
    """Construct a fresh PPO model with the hyper-parameters from config.py."""
    resolved = _get_device(device)
    print(f'[agent_brain] Building PPO on device: {resolved}')
    return PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        tensorboard_log=LOG_DIR,
        device=resolved,
        verbose=1,
    )


def load_or_create_ppo(
    vec_env: VecEnv,
    device: str = 'auto',
) -> tuple[PPO, int]:
    """Load the latest checkpoint if one exists, otherwise create a fresh PPO.

    Returns:
        (model, steps_already_trained)
    """
    resolved   = _get_device(device)
    latest_zip = os.path.join(CHECKPOINT_DIR, 'latest.zip')
    state_file = os.path.join(CHECKPOINT_DIR, 'training_state.json')

    if os.path.isfile(latest_zip):
        model = PPO.load(latest_zip, env=vec_env, device=resolved)
        steps = 0
        if os.path.isfile(state_file):
            with open(state_file) as fh:
                data = json.load(fh)
            steps = int(data.get('total_steps_trained', 0))
        print(f'[agent_brain] Resuming from checkpoint: {steps:,} steps trained')
        return model, steps

    print('[agent_brain] Starting fresh training')
    return build_ppo(vec_env, device=device), 0


def load_policy_for_play(
    checkpoint_path: str | None,
    device: str = 'auto',
) -> PPO | None:
    """Load a policy for play mode.  Returns None if no checkpoint given/found."""
    if checkpoint_path is None:
        auto = os.path.join(CHECKPOINT_DIR, 'latest.zip')
        if os.path.isfile(auto):
            checkpoint_path = auto
        else:
            return None

    if not os.path.isfile(checkpoint_path):
        print(f'[agent_brain] Checkpoint not found: {checkpoint_path}')
        return None

    resolved = _get_device(device)
    model = PPO.load(checkpoint_path, device=resolved)
    print(f'[agent_brain] Loaded policy from {checkpoint_path} on {resolved}')
    return model


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------

class TrainingStateCallback(BaseCallback):
    """Writes training_state.json and maintains a 'latest.zip' symlink.

    Fired on the same cadence as SB3's CheckpointCallback so the JSON always
    matches the most recently saved checkpoint.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq_per_env: int,
        n_envs: int,
        verbose: int = 0,
    ):
        """Initialise the callback.

        Args:
            checkpoint_dir:     Directory where checkpoints are saved.
            save_freq_per_env:  Steps per env between saves (matches CheckpointCallback).
            n_envs:             Number of parallel envs (for total-step accounting).
        """
        super().__init__(verbose)
        self.checkpoint_dir      = checkpoint_dir
        self.save_freq_per_env   = save_freq_per_env
        self.n_envs              = n_envs
        self._best_reward: float = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq_per_env != 0:
            return True

        total_steps = self.num_timesteps

        # Update best mean reward if available
        if len(self.model.ep_info_buffer) > 0:
            mean_r = float(
                sum(ep['r'] for ep in self.model.ep_info_buffer)
                / len(self.model.ep_info_buffer)
            )
            if mean_r > self._best_reward:
                self._best_reward = mean_r

        state = {
            'total_steps_trained':  total_steps,
            'date_last_trained':    datetime.utcnow().isoformat(),
            'best_reward_achieved': self._best_reward,
        }
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as fh:
            json.dump(state, fh, indent=2)

        # Copy the latest checkpoint zip to 'latest.zip'
        src = os.path.join(
            self.checkpoint_dir,
            f'rl_model_{total_steps}_steps.zip',
        )
        dst = os.path.join(self.checkpoint_dir, 'latest.zip')
        if os.path.isfile(src):
            shutil.copy2(src, dst)

        return True
