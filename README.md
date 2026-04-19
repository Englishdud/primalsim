# Primal Survival Simulation

A Python-based reinforcement-learning sandbox where primitive human agents
learn to hunt mammoths from scratch.  Physics are handled by PyBullet;
rendering by Panda3D; the RL policy is trained with PPO via stable-baselines3.

---

## What this is

Four humanoid agents start in a 1 km × 1 km procedurally generated world with
no pre-programmed hunting skills.  Through thousands of training episodes they
gradually learn to:

1. Stand upright without falling over.
2. Walk and navigate toward mammoths.
3. Pick up spears and use them.
4. Hunt and kill mammoths to survive.

Training is fully headless (no GUI) and auto-resumes every time you run it.
A separate visual mode lets you watch trained agents in action.

---

## Requirements

* Python 3.10 or newer
* A gaming PC with a dedicated NVIDIA GPU (recommended for fast training)
* ~4 GB free disk space for checkpoints and logs

---

## Installation

```bash
git clone https://github.com/Englishdud/primalsim.git
cd primalsim
pip install -r requirements.txt
```

---

## Training

```bash
python main.py train
```

Training auto-resumes from `./checkpoints/latest.zip` if it exists, so you
can stop and restart at any time.  Monitor progress in TensorBoard:

```bash
tensorboard --logdir ./logs
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 10,000,000 | Total step target (cumulative) |
| `--envs N` | 4 | Parallel SubprocVecEnv workers |
| `--device auto\|cpu\|cuda` | auto | Torch device |

---

## Watch / Play

```bash
python main.py play
```

Loads `./checkpoints/latest.zip` automatically.  Use `--random` if you want
to watch untrained agents flail around before any training:

```bash
python main.py play --random
```

Or point at a specific checkpoint:

```bash
python main.py play --checkpoint ./checkpoints/rl_model_500000_steps.zip
```

---

## Controls

| Key | Action |
|-----|--------|
| `M` | Spawn a single mammoth |
| `H` | Spawn a herd of 2-4 mammoths |
| `R` | Reset the entire simulation |
| `C` | Cycle follow camera between agents |
| `F` | Toggle free camera (WASD + mouse look) |
| `WASD` | Move (free camera mode only) |
| `Q / E` | Move down / up (free camera) |
| `ESC` | Quit |

---

## What to expect

| Steps | Behaviour |
|-------|-----------|
| 0 - 200k | Agents learn to stand and not fall over |
| 200k - 500k | Agents learn to walk and balance |
| 500k - 2M | Agents start investigating tools and moving toward mammoths |
| 2M - 5M | Full hunting behaviour emerges |
| 5M+ | Cooperative patterns and consistent kills |

Times vary considerably depending on GPU speed and random seed.  Leave
`train.py` running overnight for the best results.

---

## Project structure

```
primalsim/
├── main.py               Entry point (train / play dispatcher)
├── config.py             All tunable constants
├── environment.py        Gymnasium env wrapping PyBullet
├── humanoid.py           Humanoid body (PyBullet + Panda3D)
├── mammoth.py            Mammoth entity + scripted AI + Panda3D
├── rewards.py            Pure reward-shaping functions
├── agent_brain.py        PPO factory, checkpoint I/O, callbacks
├── train.py              Headless training loop
├── play.py               Panda3D visual sandbox
├── play_utils.py         Procedural Panda3D geometry helpers
├── hud.py                OnscreenText HUD overlay
├── sandbox_controls.py   Keyboard / mouse handlers
└── requirements.txt      Pinned dependencies
```

---

## Observation space (per agent, 75 floats)

| Index | Description |
|-------|-------------|
| 0-11 | Joint angles (radians) |
| 12-23 | Joint angular velocities |
| 24-27 | Pelvis quaternion (x, y, z, w) |
| 28-30 | Pelvis linear velocity |
| 31-33 | Pelvis angular velocity |
| 34-65 | Vision rays: 16 x (normalised distance, hit type) |
| 66-68 | Nearest mammoth relative position |
| 69-71 | Nearest spear relative position |
| 72 | Hunger (0-1) |
| 73 | Stamina (0-1) |
| 74 | Holding spear (0 or 1) |

---

## Reward table

| Event | Reward |
|-------|--------|
| Staying upright | +0.10 x uprightness |
| Moving toward mammoth | +0.20 x metres closed |
| Moving away from mammoth | -0.05 per step |
| First tool pickup | +0.50 (once per agent) |
| Landing a hit on mammoth | +2.00 |
| Killing a mammoth | +10.00 |
| Eating a carcass | +5.00 |
| Hunger below threshold | -1.00 per step |
| Wasted sprint | -0.01 per step |
| Death | -100.00 |

---

## Recording for YouTube

1. Start the visual mode: `python main.py play`
2. Open **OBS Studio**, add a **Window Capture** source pointing at the
   Panda3D window.
3. Set output resolution to 1920x1080 and record in MP4/H.264.
4. Enable game mode in Windows or set process priority to High for smooth
   capture.

---

## Tips

* **Faster training:** Use `--envs 8` if you have 8+ CPU cores.  Each worker
  is a separate Python process with its own PyBullet DIRECT server.
* **CUDA:** Install `torch` with CUDA support matching your GPU driver.  The
  simulation auto-detects and uses `cuda` when available.
* **Resume:** Every session stacks on top of the last.  Agents never forget
  what they learned in previous training runs.
* **Experiment:** All magic numbers live in `config.py`.  Adjust rewards,
  terrain, speeds, or vision range without touching any other file.
