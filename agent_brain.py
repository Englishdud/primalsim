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


def resolve_device(device: str = 'auto') -> torch.device:
    """Return the best available torch.device.

    Resolution order for 'auto':
      1. CUDA (any GPU visible to PyTorch)
      2. CPU fallback

    Raises a clear warning when CUDA is requested explicitly but unavailable.
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA requested (--device cuda) but torch.cuda.is_available() '
                'returned False.\n'
                'Check that:\n'
                '  • The CUDA-enabled PyTorch wheel is installed '
                '(pip install torch --index-url https://download.pytorch.org/whl/cu121)\n'
                '  • CUDA drivers are installed and nvidia-smi works\n'
                '  • CUDA_VISIBLE_DEVICES is not set to -1 or an empty string'
            )
        return torch.device('cuda')

    if device == 'cpu':
        return torch.device('cpu')

    # 'auto': prefer CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _get_device(device: str = 'auto') -> str:
    """Resolve 'auto' to a device string ('cuda' or 'cpu').

    Kept for backward-compat; callers that need a torch.device should use
    resolve_device() directly.
    """
    return str(resolve_device(device))


def print_gpu_info() -> None:
    """Print a detailed GPU / CUDA status block to stdout."""
    print('-' * 60)
    print('  PyTorch device info')
    print(f'  PyTorch version : {torch.__version__}')
    print(f'  CUDA available  : {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        print(f'  CUDA version    : {torch.version.cuda}')
        n = torch.cuda.device_count()
        print(f'  GPU count       : {n}')
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(
                f'  GPU {i}           : {props.name}  '
                f'({mem_gb:.1f} GB, compute {props.major}.{props.minor})'
            )
        cur = torch.cuda.current_device()
        print(f'  Active GPU      : {cur} – {torch.cuda.get_device_name(cur)}')
        # Enable cuDNN auto-tuner for fixed-size inputs (gives ~10-20% speedup)
        torch.backends.cudnn.benchmark = True
        print('  cuDNN benchmark : enabled')
    else:
        print('  Training will run on CPU (slower).')
        print('  Install the CUDA wheel: '
              'pip install torch --index-url https://download.pytorch.org/whl/cu121')
    print('-' * 60)


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
    dev = resolve_device(device)
    print(f'[agent_brain] Building fresh PPO on device: {dev}')
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        tensorboard_log=LOG_DIR,
        device=dev,
        verbose=1,
    )
    _verify_model_device(model, dev)
    return model


def load_or_create_ppo(
    vec_env: VecEnv,
    device: str = 'auto',
) -> tuple[PPO, int]:
    """Load the latest checkpoint if one exists, otherwise create a fresh PPO.

    Returns:
        (model, steps_already_trained)
    """
    dev        = resolve_device(device)
    latest_zip = os.path.join(CHECKPOINT_DIR, 'latest.zip')
    state_file = os.path.join(CHECKPOINT_DIR, 'training_state.json')

    if os.path.isfile(latest_zip):
        print(f'[agent_brain] Loading checkpoint {latest_zip} → device {dev}')
        model = PPO.load(latest_zip, env=vec_env, device=dev)
        _verify_model_device(model, dev)
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

    dev   = resolve_device(device)
    model = PPO.load(checkpoint_path, device=dev)
    _verify_model_device(model, dev)
    print(f'[agent_brain] Loaded policy from {checkpoint_path} on {dev}')
    return model


def _verify_model_device(model: PPO, expected: torch.device) -> None:
    """Confirm the policy network parameters are on the expected device.

    Prints a warning if there is a mismatch (e.g. checkpoint was on CPU but
    CUDA was requested and the tensors were not moved).
    """
    try:
        actual = next(model.policy.parameters()).device
        if actual.type != expected.type:
            print(
                f'[agent_brain] WARNING: model parameters are on {actual} '
                f'but {expected} was requested.  Forcing move…'
            )
            model.policy.to(expected)
            actual = next(model.policy.parameters()).device
        print(f'[agent_brain] Policy network confirmed on device: {actual}')
    except StopIteration:
        pass  # no parameters – shouldn't happen with MlpPolicy


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
