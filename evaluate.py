"""Deterministic evaluation of a trained policy: one full episode (15 s) in N parallel envs.

Reports episode return (same reward as training), forward distance, average speed and fall rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from ant_env import MyantGenesisEnv
from ant_model import write_ant_xml
from train import ppo_runner_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg_length", type=float, required=True)
    parser.add_argument("--foot_length", type=float, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--init_noise", type=float, default=0.1)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    xml = write_ant_xml(args.leg_length, args.foot_length, os.path.splitext(args.out)[0] + ".xml")
    env = MyantGenesisEnv(xml, args.foot_length, num_envs=args.num_envs, init_noise=args.init_noise)
    runner = OnPolicyRunner(env, ppo_runner_cfg(0), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    n = args.num_envs
    ret = torch.zeros(n, device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    x0 = env.torso_position[:, 0].clone()
    dist = torch.zeros(n, device=env.device)
    obs = env.get_observations()
    for _ in range(env.max_episode_length - 1):
        with torch.inference_mode():
            actions = policy(obs)
        x_before = env.torso_position[:, 0].clone()
        obs, rew, dones, extras = env.step(actions)
        ret += torch.where(alive, rew, 0.0)
        died = dones & ~extras["time_outs"]
        # distance up to the end of the first episode (position before reset)
        dist = torch.where(alive & dones, x_before - x0, dist)
        fell |= alive & died
        alive &= ~dones
    dist = torch.where(alive, env.torso_position[:, 0] - x0, dist)
    duration = (env.max_episode_length - 1) * env.step_dt
    result = {
        "leg_length": args.leg_length,
        "foot_length": args.foot_length,
        "checkpoint": args.checkpoint,
        "init_noise": args.init_noise,
        "num_envs": n,
        "episode_return_mean": ret.mean().item(),
        "episode_return_std": ret.std().item(),
        "distance_mean_m": dist.mean().item(),
        "speed_mean_mps": (dist / duration).mean().item(),
        "fall_rate": fell.float().mean().item(),
    }
    print(json.dumps(result))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
