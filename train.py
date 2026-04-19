"""Headless training entry point for the Primal Survival Simulation.

Runs fully in PyBullet DIRECT mode (no GUI, no Panda3D).
Automatically resumes from the latest checkpoint if one exists.

Usage:
    python train.py
    python train.py --steps 5000000 --envs 4 --device cuda
"""

import argparse
import os
import sys

from stable_baselines3.common.callbacks import CheckpointCallback

from config import (
    CHECKPOINT_DIR, LOG_DIR, CHECKPOINT_FREQ, N_ENVS,
)
from agent_brain import (
    build_vec_env,
    load_or_create_ppo,
    TrainingStateCallback,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description='Train the primal survival simulation with PPO.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--steps', type=int, default=10_000_000,
        help='Total training steps target (across all envs).',
    )
    parser.add_argument(
        '--envs', type=int, default=N_ENVS,
        help='Number of parallel SubprocVecEnv workers.',
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Torch device: auto detects CUDA and falls back to CPU.',
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for headless training."""
    args = parse_args()

    # Ensure output directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print('=' * 60)
    print('  Primal Survival Simulation – PPO Training')
    print('=' * 60)
    print(f'  Parallel envs : {args.envs}')
    print(f'  Target steps  : {args.steps:,}')
    print(f'  Device        : {args.device}')
    print(f'  Checkpoint dir: {CHECKPOINT_DIR}')
    print(f'  TensorBoard   : {LOG_DIR}')
    print('=' * 60)

    # Build parallel environments
    print('[train] Spawning parallel environments...')
    vec_env = build_vec_env(n_envs=args.envs)

    # Load or create PPO model
    model, steps_trained = load_or_create_ppo(vec_env, device=args.device)

    remaining = args.steps - steps_trained
    if remaining <= 0:
        print(f'[train] Already at {steps_trained:,} steps – nothing to do.')
        vec_env.close()
        return

    print(f'[train] Training for {remaining:,} additional steps...')

    # save_freq for CheckpointCallback is per-env steps
    save_freq_per_env = max(1, CHECKPOINT_FREQ // args.envs)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=CHECKPOINT_DIR,
        name_prefix='rl_model',
        verbose=1,
    )
    state_cb = TrainingStateCallback(
        checkpoint_dir=CHECKPOINT_DIR,
        save_freq_per_env=save_freq_per_env,
        n_envs=args.envs,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_cb, state_cb],
            reset_num_timesteps=False,    # preserve step counter for resume
            tb_log_name='ppo_primal',
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print('\n[train] Training interrupted by user.')
    finally:
        # Save final checkpoint regardless of how training ended
        final_path = os.path.join(CHECKPOINT_DIR, 'latest.zip')
        model.save(final_path)
        print(f'[train] Final model saved to {final_path}')
        vec_env.close()

    print('[train] Done.')


if __name__ == '__main__':
    main()
