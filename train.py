"""Train MyAnt (given leg/foot length) with rsl_rl PPO on Genesis.

Equivalent of `isaaclab.sh -p scripts/rsl_rl/train.py --task=Template-Myant-Train-Direct-v0`.
Prints rsl_rl's standard log (including "Mean reward:") to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from ant_env import MyantGenesisEnv
from ant_model import DEFAULT_FOOT_LENGTH, DEFAULT_LEG_LENGTH, write_ant_xml


def ppo_runner_cfg(seed: int) -> dict:
    """Same values as MyAnt_train agents/rsl_rl_ppo_cfg.py (PPORunnerCfg)."""
    return {
        "seed": seed,
        "num_steps_per_env": 16,
        "save_interval": 100000,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [32, 32],
            "critic_hidden_dims": [32, 32],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg_length", type=float, default=DEFAULT_LEG_LENGTH)
    parser.add_argument("--foot_length", type=float, default=DEFAULT_FOOT_LENGTH)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)

    xml = write_ant_xml(args.leg_length, args.foot_length, os.path.join(args.log_dir, "myant.xml"))
    env = MyantGenesisEnv(xml, args.foot_length, num_envs=args.num_envs)

    runner = OnPolicyRunner(env, ppo_runner_cfg(args.seed), log_dir=args.log_dir, device="cuda:0")
    t0 = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {time.time() - t0:.1f}s")
    runner.save(os.path.join(args.log_dir, "model_final.pt"))


if __name__ == "__main__":
    main()
