"""Entry point for the Primal Survival Simulation.

Dispatches to either headless training (train.py) or the visual play mode
(play.py) based on the first positional argument.

Usage:
    python main.py train [--steps N] [--envs N] [--device auto|cpu|cuda]
    python main.py play  [--checkpoint PATH] [--random]
    python main.py play  --random
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with train/play sub-commands."""
    parser = argparse.ArgumentParser(
        prog='primalsim',
        description='Primal Human Survival Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                     # train with defaults
  python main.py train --steps 5000000     # shorter run
  python main.py play                      # watch latest checkpoint
  python main.py play --random             # random policy (no checkpoint)
  python main.py play --checkpoint ./checkpoints/rl_model_1000000_steps.zip
        """,
    )

    subs = parser.add_subparsers(dest='command', required=True)

    # ── train ──────────────────────────────────────────────────────────
    train_p = subs.add_parser(
        'train',
        help='Run headless PPO training (auto-resumes from last checkpoint).',
    )
    train_p.add_argument(
        '--steps', type=int, default=10_000_000,
        help='Total training step target (cumulative across all sessions).',
    )
    train_p.add_argument(
        '--envs', type=int, default=None,
        help='Number of parallel SubprocVecEnv workers (default: N_ENVS from config).',
    )
    train_p.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Torch device for the policy network.',
    )

    # ── play ───────────────────────────────────────────────────────────
    play_p = subs.add_parser(
        'play',
        help='Launch the Panda3D visual simulation.',
    )
    play_p.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to a specific .zip policy file (omit to use latest).',
    )
    play_p.add_argument(
        '--random', action='store_true',
        help='Ignore all checkpoints and use a random policy.',
    )

    return parser


def _run_train(args: argparse.Namespace) -> None:
    """Delegate to train.py's main() with overridden sys.argv."""
    argv = ['train', '--steps', str(args.steps), '--device', args.device]
    if args.envs is not None:
        argv += ['--envs', str(args.envs)]
    sys.argv = argv
    from train import main as train_main
    train_main()


def _run_play(args: argparse.Namespace) -> None:
    """Delegate to play.py's main() with overridden sys.argv."""
    argv = ['play']
    if args.checkpoint:
        argv += ['--checkpoint', args.checkpoint]
    if args.random:
        argv.append('--random')
    sys.argv = argv
    from play import main as play_main
    play_main()


def main() -> None:
    """Parse arguments and dispatch to the correct sub-module."""
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == 'train':
        _run_train(args)
    elif args.command == 'play':
        _run_play(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
