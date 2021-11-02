import numpy as np
import time

from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid

dyn_grid=DynamicMiniGrid()
prob_dict = {
    "alter_start_pos": 0,
    "alter_goal_pos": 0,
    "wall": 0,
    "lava": 0.5,
    "sand": 0.5,
}
n_alterations = 100
sleep_time = 0.5

for _ in range(n_alterations):
    dyn_grid.alter(prob_dict)
    dyn_grid.render()
    time.sleep(sleep_time)
