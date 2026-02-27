# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class MirpoMarlEnvCfg(DirectMARLEnvCfg):
    # --------------------------------------------------------
    # General
    # --------------------------------------------------------
    decimation = 2
    episode_length_s = 20.0

    possible_agents = ["agent1", "agent2"]

    # Each agent: Fx, Fy
    action_spaces = {
        "agent1": 2,
        "agent2": 2,
    }

    # Observation: x, y, goal_x, goal_y
    observation_spaces = {
        "agent1": 4,
        "agent2": 4,
    }

    state_space = -1

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------
    sim = SimulationCfg(
        dt=1 / 50,
        render_interval=decimation,
    )

    # --------------------------------------------------------
    # Maze definition
    # --------------------------------------------------------
    cell_size = 1.0
    wall_height = 1.0
    wall_thickness = 1.0

    # Updated to explicitly map starts (r1, r2) and goals (g1, g2)
    # maze = [
    #     [1, 1,    1,    1, 1],
    #     [1, "r1", 0, 0, 1],
    #     [1, 1,    1,    "g", 1],
    #     [1, "r2", 0, 0, 1],
    #     [1, 1,    1,    1, 1],
    # ]
    maze = [
        [1, 1, 1, 1],
        [1, "r1", 0, 1],
        [1, 1, "g", 1],
        [1, "r2", 0, 1],
        [1, 1, 1, 1],
    ]

    env_length = cell_size * max(len(maze), len(maze[0]))
    
    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------
    scene = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=env_length + 1.0,
        replicate_physics=True,
    )

    

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------
    agent_radius = 0.15
    agent_mass = 1.0
    action_scale = 1.0

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------
    goal_threshold = 0.3
    rew_goal = 10.0
    rew_step = -0.01