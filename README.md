# JAX RLHF Pipeline

A lightweight, modular RLHF (Reinforcement Learning from Human Feedback) pipeline built entirely in JAX. Implements the full post-training stack: SFT, reward modeling, PPO, DPO, and GRPO.

## Pipeline Overview

```
Stage 1: SFT (Supervised Fine-Tuning)
    |
    v
Stage 2: Reward Model Training
    |
    +---> Stage 3A: PPO  (Proximal Policy Optimization)
    +---> Stage 3B: DPO  (Direct Preference Optimization)
    +---> Stage 3C: GRPO (Group Relative Policy Optimization)
```
## Progress
*   02/13/2026: Set up the backbone model (GPT2)
