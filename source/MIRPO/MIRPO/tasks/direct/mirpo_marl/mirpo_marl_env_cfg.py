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
    episode_length_s = 10.0

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

    maze_layout = [
        [1, 1,   1,   1, 1],
        [1, "f", "f", 0, 1],
        [1, 0,   1, "w", 1],
        [1, 0, 0, "w", 1],
        [1, 1,   1,   1, 1],
    ]

    maze_config = [
        [1, 1,    1,   1, 1],
        [1, 0, 0, "g", 1],
        [1, "r1",    1,   0, 1],
        [1, 0, "r2",   0, 1],
        [1, 1,    1,   1, 1],
    ]

    # 1. Validate that both layers have the same number of rows
    assert len(maze_layout) == len(maze_config), "Layout and Config have different row counts!"

    num_rows = len(maze_layout)
    num_cols = len(maze_layout[0])

    # 2. Validate that every row is the exact same length (no jagged grids)
    for i in range(num_rows):
        assert len(maze_layout[i]) == num_cols, f"maze_layout row {i} length mismatch!"
        assert len(maze_config[i]) == num_cols, f"maze_config row {i} length mismatch!"

    # 3. Calculate exact dimensions (Do not use max(), calculate X and Y separately)
    env_length_x = cell_size * num_rows
    env_length_y = cell_size * num_cols

    # 4. Check if there exists multiple g
    assert sum(row.count("g") for row in maze_config) == 1, "There should be exactly one goal (g) in the maze_config!"

    # 5. Check if there exists one r1 and r2
    assert sum(row.count("r1") for row in maze_config) == 1, "There should be exactly one agent start (r1) in the maze_config!"
    assert sum(row.count("r2") for row in maze_config) == 1, "There should be exactly one agent start (r2) in the maze_config!"

    # 6. Check if the maze has hole in its boundary
    for i in range(num_rows):
        assert maze_layout[i][0] == 1 and maze_layout[i][-1] == 1, f"Boundary wall missing at row {i}!"
        assert maze_config[i][0] == 1 and maze_config[i][-1] == 1, f"Boundary wall missing at row {i}!"
    for j in range(num_cols):
        assert maze_layout[0][j] == 1 and maze_layout[-1][j] == 1, f"Boundary wall missing at column {j}!"
        assert maze_config[0][j] == 1 and maze_config[-1][j] == 1, f"Boundary wall missing at column {j}!"
    
    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------
    scene = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=max(env_length_x, env_length_y) + 1.0,
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
    rew_step = 0.1 # -0.01