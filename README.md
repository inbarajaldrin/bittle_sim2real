# Bittle Sim2Real

This project trains and tests a quadruped robot "Bittle" from Petoi in Isaac Lab using reinforcement learning. It supports simulation on flat terrain with video recording, pretrained checkpoints, and headless execution.

---

## Setup

### 1. Add the Bittle USD File
Place your custom `bittle_edit.usd` file in your Omniverse Nucleus Library:

```bash
omniverse://localhost/Library/bittle_edit.usd
```

### 2. Add the Config Folder
Copy your Bittle config directory to the IsaacLab task config path:

```bash
cd ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
# Then place the `bittle` folder here
```

---

## Training

### Train on Flat Terrain (headless + record video)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Bittle-v0 --headless --video
```

- Videos will be saved to:
  ```
  logs/sb3/Isaac-Velocity-Flat-Bittle-v0/<run-dir>/videos/train
  ```

### Train From Pretrained Model

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Bittle-v0 \
  --num_envs=50 \
  --checkpoint=logs/rsl_rl/bittle_flat/<run-dir>/model_2999.pt
```

---

## Playing the Trained Model

### Play with a Specific Checkpoint

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Bittle-v0 \
  --num_envs=32 \
  --checkpoint=logs/rsl_rl/bittle_flat/<run-dir>/model_1400.pt
```

### Play From Most Recent Model

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Bittle-v0 \
  --num_envs=50 \
  --checkpoint=logs/rsl_rl/bittle_flat/<run-dir>/model_2999.pt
```

---

## Notes

- For rendering, **remove the `--headless` flag** during training.
- Replace `<run-dir>` with your specific run directory timestamp (e.g., `2025-04-11_00-13-37`).
