"""Genesis port of Isaac Lab's LocomotionEnv + MyantTrainEnvCfg (Template-Myant-Train-Direct-v0).

Rewards, observations (36-dim), actions (8-dim effort) and termination follow
isaaclab_tasks/direct/locomotion/locomotion_env.py with the MyAnt config values.
"""

from __future__ import annotations

import math

import genesis as gs
import torch
from tensordict import TensorDict

from ant_model import INIT_FOOT_ANGLE, spawn_height


class MyantTrainEnvCfg:
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 36
    dt = 1 / 120
    gravity = (0.0, 0.0, -9.81)  # Earth

    joint_gears = [15.0] * 8
    heading_weight = 0.5
    up_weight = 0.1
    energy_cost_scale = 0.05
    actions_cost_scale = 0.005
    alive_reward_scale = 0.5
    dof_vel_scale = 0.2
    death_cost = -2.0
    termination_height = 0.25
    angular_velocity_scale = 1.0
    ground_friction = 1.0


JOINT_NAMES = [
    "front_left_leg", "front_right_leg", "left_back_leg", "right_back_leg",
    "front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot",
]


# --- quaternion helpers (w, x, y, z) -------------------------------------------------
def quat_rotate(q, v):
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q, v):
    q_conj = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    return quat_rotate(q_conj, v)


def get_euler_xyz(q):
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll % (2 * math.pi), pitch % (2 * math.pi), yaw % (2 * math.pi)


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class MyantGenesisEnv:
    """rsl_rl compatible vectorized env."""

    def __init__(self, xml_path, foot_length, num_envs=512, cfg=MyantTrainEnvCfg, camera=False, device="cuda", init_noise=0.0):
        self.cfg = cfg
        self.init_noise = init_noise  # evaluation only: uniform noise [rad] on initial joint angles
        self.num_envs = num_envs
        self.num_actions = cfg.action_space
        self.device = torch.device(device)
        self.step_dt = cfg.dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=1, gravity=cfg.gravity),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                enable_joint_limit=True,
                max_collision_pairs=64,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
            show_viewer=False,
            show_FPS=False,
        )
        self.scene.add_entity(
            gs.morphs.Plane(), material=gs.materials.Rigid(friction=cfg.ground_friction)
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(xml_path)))
        self.cam = None
        if camera:
            self.cam = self.scene.add_camera(res=(640, 480), pos=(3.0, -4.0, 2.5), lookat=(0, 0, 0.5), fov=45, GUI=False)
        self.scene.build(n_envs=num_envs)

        self.dofs_idx = [self.robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
        lower, upper = self.robot.get_dofs_limit(self.dofs_idx)
        self.dof_lower = lower[0].to(self.device) if lower.dim() > 1 else lower.to(self.device)
        self.dof_upper = upper[0].to(self.device) if upper.dim() > 1 else upper.to(self.device)

        n = num_envs
        self.joint_gears = torch.tensor(cfg.joint_gears, device=self.device)
        self.init_pos = torch.tensor([0.0, 0.0, spawn_height(foot_length)], device=self.device)
        self.init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.init_dof = torch.tensor([INIT_FOOT_ANGLE.get(j, 0.0) for j in JOINT_NAMES], device=self.device)
        self.targets = torch.tensor([1000.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        self.basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        self.basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)

        self.actions = torch.zeros(n, self.num_actions, device=self.device)
        self.potentials = torch.zeros(n, device=self.device)
        self.prev_potentials = torch.zeros(n, device=self.device)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=self.device)
        self.extras = {}

        self._reset_idx(torch.arange(n, device=self.device))
        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        k = len(env_ids)
        self.robot.set_pos(self.init_pos.repeat(k, 1), envs_idx=env_ids, zero_velocity=True)
        self.robot.set_quat(self.init_quat.repeat(k, 1), envs_idx=env_ids, zero_velocity=True)
        dof = self.init_dof.repeat(k, 1)
        if self.init_noise > 0:
            dof = dof + (torch.rand_like(dof) * 2 - 1) * self.init_noise
        self.robot.set_dofs_position(dof, self.dofs_idx, envs_idx=env_ids, zero_velocity=True)
        self.robot.zero_all_dofs_velocity(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        to_target = self.targets[env_ids] - self.init_pos
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, dim=-1) / self.cfg.dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def _compute_intermediate_values(self):
        self.torso_position = self.robot.get_pos().to(self.device)
        self.torso_rotation = self.robot.get_quat().to(self.device)
        velocity = self.robot.get_vel().to(self.device)
        ang_velocity = self.robot.get_ang().to(self.device)
        self.dof_pos = self.robot.get_dofs_position(self.dofs_idx).to(self.device)
        self.dof_vel = self.robot.get_dofs_velocity(self.dofs_idx).to(self.device)

        to_target = self.targets - self.torso_position
        to_target[:, 2] = 0.0
        q = self.torso_rotation
        up_vec = quat_rotate(q, self.basis_vec1)
        heading_vec = quat_rotate(q, self.basis_vec0)
        self.up_proj = up_vec[:, 2]
        self.heading_proj = (heading_vec * torch.nn.functional.normalize(to_target, dim=-1)).sum(-1)

        self.vel_loc = quat_rotate_inverse(q, velocity)
        self.angvel_loc = quat_rotate_inverse(q, ang_velocity)
        self.roll, self.pitch, self.yaw = get_euler_xyz(q)
        walk_target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        self.angle_to_target = walk_target_angle - self.yaw

        self.dof_pos_scaled = (2.0 * self.dof_pos - self.dof_upper - self.dof_lower) / (self.dof_upper - self.dof_lower)

        self.prev_potentials[:] = self.potentials
        self.potentials = -torch.norm(to_target, dim=-1) / self.cfg.dt

    def _get_observations(self):
        c = self.cfg
        obs = torch.cat(
            (
                self.torso_position[:, 2:3],
                self.vel_loc,
                self.angvel_loc * c.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * c.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def _get_rewards(self, terminated):
        c = self.cfg
        heading_reward = torch.where(
            self.heading_proj > 0.8,
            torch.full_like(self.heading_proj, c.heading_weight),
            c.heading_weight * self.heading_proj / 0.8,
        )
        up_reward = torch.where(self.up_proj > 0.93, c.up_weight, 0.0)
        actions_cost = torch.sum(self.actions**2, dim=-1)
        electricity_cost = torch.sum(torch.abs(self.actions * self.dof_vel * c.dof_vel_scale), dim=-1)
        dof_at_limit_cost = torch.sum(self.dof_pos_scaled > 0.98, dim=-1)
        progress_reward = self.potentials - self.prev_potentials
        total = (
            progress_reward
            + c.alive_reward_scale
            + up_reward
            + heading_reward
            - c.actions_cost_scale * actions_cost
            - c.energy_cost_scale * electricity_cost
            - dof_at_limit_cost
        )
        total = torch.where(terminated, torch.full_like(total, c.death_cost), total)
        return torch.nan_to_num(total, nan=c.death_cost)

    # ------------------------------------------------------------------
    def get_observations(self):
        return self._get_observations()

    def step(self, actions):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        forces = self.cfg.action_scale * self.joint_gears * self.actions
        for _ in range(self.cfg.decimation):
            self.robot.control_dofs_force(forces, self.dofs_idx)
            self.scene.step()

        self.episode_length_buf += 1
        self._compute_intermediate_values()
        died = (self.torso_position[:, 2] < self.cfg.termination_height) | ~torch.isfinite(self.torso_position).all(-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        rewards = self._get_rewards(died)
        dones = died | time_out

        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)
            self._compute_intermediate_values()
            # potentials of reset envs must not produce a progress jump on the next step
            self.prev_potentials[reset_ids] = self.potentials[reset_ids]
        obs = self._get_observations()
        self.extras = {"time_outs": time_out, "log": {}}
        return obs, rewards, dones, self.extras
