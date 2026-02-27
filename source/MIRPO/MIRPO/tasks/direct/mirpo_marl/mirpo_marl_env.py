# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    PreviewSurfaceCfg,
)

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from .mirpo_marl_env_cfg import MirpoMarlEnvCfg


def define_markers(goal_radius: float) -> VisualizationMarkers:
    """Define persistent USD markers for visual debugging."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MirpoMarkers",
        markers={
            # Index 0: Red Pad for the Goal (Radius matches the termination threshold)
            "goal_pad": sim_utils.CylinderCfg(
                radius=goal_radius,
                height=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            # Index 1: Blue Arrow pointing along velocity (Thinner and longer)
            "velocity_arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.1, 0.5), # (Height, Width, Length)
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
            ),
            # Index 2: Red Arrow pointing to goal (Thinner and longer)
            "goal_dir_arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.1, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class MirpoMarlEnv(DirectMARLEnv):
    cfg: MirpoMarlEnvCfg

    def __init__(self, cfg: MirpoMarlEnvCfg, render_mode=None, **kwargs):
        self.agent_prims: dict[str, RigidObject] = {}
        
        # Define the exact radius that counts as "reaching the goal"
        self.goal_threshold = cfg.goal_threshold
        
        super().__init__(cfg, render_mode, **kwargs)
        self._parse_maze()

    # --------------------------------------------------------
    # Scene setup
    # --------------------------------------------------------
    def _setup_scene(self):
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        self._parse_maze()

        wall_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.cell_size, self.cfg.cell_size, self.cfg.wall_height),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        )

        for i, pos in enumerate(self.wall_positions):
            wall_cfg.func(
                f"/World/envs/env_0/Wall_{i}",
                wall_cfg,
                translation=(pos[0], pos[1], self.cfg.wall_height / 2),
            )

        for agent_name in self.cfg.possible_agents:
            sphere_cfg = sim_utils.SphereCfg(
                radius=self.cfg.agent_radius,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.5, 
                    angular_damping=0.5,
                ),
                mass_props=MassPropertiesCfg(mass=self.cfg.agent_mass),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            )
            sphere_cfg.func(
                f"/World/envs/env_0/{agent_name}", 
                sphere_cfg, 
                translation=(0.0, 0.0, self.cfg.agent_radius)
            )

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

        for agent_name in self.cfg.possible_agents:
            ro_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{agent_name}",
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            self.agent_prims[agent_name] = RigidObject(cfg=ro_cfg)
            self.scene.rigid_objects[agent_name] = self.agent_prims[agent_name]

        # --- Initialize Visualization Markers ---
        # Pass the threshold to dictate the size of the visual red pad
        self.visualization_markers = define_markers(goal_radius=self.goal_threshold)
        
        all_envs = torch.arange(self.num_envs, device=self.device)
        num_agents = len(self.cfg.possible_agents)
        self.marker_indices = torch.cat([
            torch.full_like(all_envs, 0), 
            torch.full((self.num_envs * num_agents,), 1, device=self.device, dtype=torch.long), 
            torch.full((self.num_envs * num_agents,), 2, device=self.device, dtype=torch.long), 
        ])

    # --------------------------------------------------------
    # Maze parsing
    # --------------------------------------------------------
    def _parse_maze(self):
        self.wall_positions = []
        self.agent_starts = {}
        self.shared_goal_position = None 

        for i, row in enumerate(self.cfg.maze):
            for j, cell in enumerate(row):
                pos = [i * self.cfg.cell_size, j * self.cfg.cell_size]

                if cell == 1:
                    self.wall_positions.append(pos)
                elif isinstance(cell, str):
                    if cell.startswith("r"):
                        idx = cell[1:]
                        self.agent_starts[f"agent{idx}"] = pos
                    elif cell == "g":
                        self.shared_goal_position = torch.tensor(pos, device=self.device)

        if self.shared_goal_position is None:
            raise ValueError("No goal ('g') found in the maze configuration!")

    # --------------------------------------------------------
    # Visualization Logic
    # --------------------------------------------------------
    def _visualize_markers(self):
        """Updates the locations and rotations of goals and arrows using Isaac Lab's Marker API."""
        num_envs = self.num_envs
        up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # 1. Goal Pad Tensors
        goal_pos_local = self.shared_goal_position.unsqueeze(0).repeat(num_envs, 1)
        goal_pos_world = goal_pos_local + self.scene.env_origins[:, :2]
        
        goal_locs = torch.cat([goal_pos_world, torch.full((num_envs, 1), 0.01, device=self.device)], dim=-1)
        goal_rots = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)

        # 2. Agent Arrows Tensors
        vel_locs, vel_rots = [], []
        dir_locs, dir_rots = [], []
        
        # Calculate a single vertical offset so both arrows originate from the exact same point
        arrow_z_offset = torch.tensor([0.0, 0.0, self.cfg.agent_radius + 0.3], device=self.device)

        for agent in self.cfg.possible_agents:
            pos_w = self.agent_prims[agent].data.root_pos_w
            vel_w = self.agent_prims[agent].data.root_lin_vel_w
            
            # --- Velocity Arrow (Blue) ---
            vel_norm = torch.norm(vel_w[:, :2], dim=-1) + 1e-5
            vel_yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])
            vel_quat = math_utils.quat_from_angle_axis(vel_yaw.view(-1, 1), up_dir).view(-1, 4)
            
            vel_locs.append(pos_w + arrow_z_offset)
            vel_rots.append(vel_quat)
            
            # --- Goal Direction Arrow (Red) ---
            to_goal = goal_pos_world - pos_w[:, :2]
            goal_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
            dir_quat = math_utils.quat_from_angle_axis(goal_yaw.view(-1, 1), up_dir).view(-1, 4)
            
            # Use the EXACT same offset so the arrows overlap
            dir_locs.append(pos_w + arrow_z_offset) 
            dir_rots.append(dir_quat)

        vel_locs = torch.cat(vel_locs, dim=0)
        vel_rots = torch.cat(vel_rots, dim=0)
        dir_locs = torch.cat(dir_locs, dim=0)
        dir_rots = torch.cat(dir_rots, dim=0)

        # 3. Stack everything (Strict Order: Goals, Vels, Dirs)
        all_locs = torch.cat([goal_locs, vel_locs, dir_locs], dim=0)
        all_rots = torch.cat([goal_rots, vel_rots, dir_rots], dim=0)

        # 4. Render
        self.visualization_markers.visualize(all_locs, all_rots, marker_indices=self.marker_indices)

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        
    def _apply_action(self) -> None:
        for agent, action in self.actions.items():
            force = torch.zeros((self.num_envs, 1, 3), device=self.device)
            force[:, 0, 0:2] = action * self.cfg.action_scale
            
            self.agent_prims[agent].permanent_wrench_composer.set_forces_and_torques(
                forces=force, 
                torques=torch.zeros_like(force) 
            )

    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {}
        for agent in self.cfg.possible_agents:
            pos_w = self.agent_prims[agent].data.root_pos_w
            pos_local = pos_w - self.scene.env_origins
            pos_2d = pos_local[:, :2]
            
            goal_2d = self.shared_goal_position.unsqueeze(0).expand(self.num_envs, 2)
            obs[agent] = torch.cat([pos_2d, goal_2d], dim=-1)
            
        self._visualize_markers()
        return obs

    # --------------------------------------------------------
    # Rewards
    # --------------------------------------------------------
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}
        for agent in self.cfg.possible_agents:
            pos_w = self.agent_prims[agent].data.root_pos_w
            pos_local = pos_w - self.scene.env_origins
            pos_2d = pos_local[:, :2]
            
            goal_2d = self.shared_goal_position.unsqueeze(0).expand(self.num_envs, 2)
            dist = torch.norm(pos_2d - goal_2d, dim=-1)
            
            # Give a large reward if they are inside the threshold, otherwise a step penalty
            reached_goal = dist < self.goal_threshold
            
            rewards[agent] = torch.where(
                reached_goal, 
                torch.full_like(dist, self.cfg.rew_goal), 
                -dist + self.cfg.rew_step
            )
            
        return rewards

    # --------------------------------------------------------
    # Dones
    # --------------------------------------------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {}
        time_outs = {}
        
        for agent in self.cfg.possible_agents:
            pos_w = self.agent_prims[agent].data.root_pos_w
            pos_local = pos_w - self.scene.env_origins
            pos_2d = pos_local[:, :2]
            
            goal_2d = self.shared_goal_position.unsqueeze(0).expand(self.num_envs, 2)
            dist = torch.norm(pos_2d - goal_2d, dim=-1)
            
            # Determine if THIS specific agent reached the goal pad
            reached_goal = dist < self.goal_threshold
            
            # Terminate if the agent reaches the goal OR the time runs out
            terminated[agent] = time_out | reached_goal
            time_outs[agent] = time_out.clone()
            
        return terminated, time_outs

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.agent_prims["agent1"]._ALL_INDICES

        super()._reset_idx(env_ids)
        
        for agent, prim in self.agent_prims.items():
            start_pos = torch.tensor(self.agent_starts[agent], device=self.device)
            root_state = prim.data.default_root_state[env_ids].clone()
            
            root_state[:, :2] = start_pos
            root_state[:, 2] = self.cfg.agent_radius
            root_state[:, :3] += self.scene.env_origins[env_ids]
            
            prim.write_root_state_to_sim(root_state, env_ids)
            
        self._visualize_markers()