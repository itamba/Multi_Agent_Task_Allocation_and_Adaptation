# Legacy Code

Archived code from earlier development stages, preserved for reference.

## Contents

- **`train_full_dqn.py`** — Original training script using DQN (Deep Q-Network) with epsilon-greedy exploration. Replaced by MAPPO in `train_full.py`.

- **`dqn_training/`** — DQN training components:
  - `buffer.py` — Replay buffer for experience replay
  - `trainer.py` — DQNTrainer with target network updates
  - `training_utils.py` — DQN-specific utilities

## Why Legacy?

The project evolved from DQN (single-agent, value-based) to MAPPO (multi-agent, policy-based with CTDE). The DQN approach was replaced because:
- MAPPO naturally handles multi-agent coordination
- Centralized critic enables better credit assignment
- On-policy learning is more stable for our imitation reward structure
