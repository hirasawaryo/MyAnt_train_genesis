"""Roll out a trained policy and save a short walking video (mp4).

Equivalent of `scripts/rsl_rl/play.py --video --video_length ...`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
import numpy as np
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
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video_length", type=int, default=480, help="env steps (60 steps = 1 s)")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")
    xml = write_ant_xml(args.leg_length, args.foot_length, os.path.splitext(args.video)[0] + ".xml")
    env = MyantGenesisEnv(xml, args.foot_length, num_envs=1, camera=True)

    runner = OnPolicyRunner(env, ppo_runner_cfg(0), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()
    cam = env.cam
    size = max(args.leg_length + args.foot_length, 0.8)
    cam.start_recording(save_to_filename=args.video, fps=60)
    start_x = env.torso_position[0, 0].item()
    for _ in range(args.video_length):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        p = env.torso_position[0].cpu().numpy()
        cam.set_pose(pos=(p[0] + 1.5 * size, p[1] - 3.5 * size, 1.2 + 1.2 * size), lookat=(p[0], p[1], 0.3))
        cam.render()
    distance = env.torso_position[0, 0].item() - start_x
    cam.stop_recording()
    print(f"Video saved: {args.video}  (x displacement in last episode segment: {distance:.2f} m)")


if __name__ == "__main__":
    main()
