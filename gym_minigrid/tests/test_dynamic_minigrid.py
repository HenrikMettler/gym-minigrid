import pytest
import numpy as np

from pytest import fixture
from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid
from gym_minigrid.minigrid import Wall


@fixture
def dyn_grid():
    return DynamicMiniGrid()


@fixture
def prob_dict():
    return {
    "alter_start_pos": 0.2,
    "alter_goal_pos": 0.2,
    "wall": 0.2,
    "lava": 0.2,
    "sand": 0.2,
}


def count_grid_differences(grid1, grid2):
    counter = 0
    for idx, el in enumerate(grid1):
        if el is None and grid2[idx] is not None:
            counter += 1
        if el is not None:
            if grid2[idx] is None or el.type != grid2[idx].type:
                counter += 1
    return counter


def test_alter_changes_single_grid(dyn_grid, prob_dict):
    dyn_grid_copy = DynamicMiniGrid()

    dyn_grid.alter(prob_dict)

    grid_diff_counter = count_grid_differences(dyn_grid.grid.grid, dyn_grid_copy.grid.grid)

    assert grid_diff_counter == 1


def test_alter_call_repeatedly(n_alter_calls=3):
    # n_alter can't be too high, because alterations can be reverted
    prob_dict = {
    "alter_start_pos": 0,
    "alter_goal_pos": 0,
    "wall": 0.5,
    "lava": 0.5,
    "sand": 0,
    }   # add objects to ensure single change
    dyn_grid = DynamicMiniGrid()
    dyn_grid_copy = DynamicMiniGrid()

    for _ in range(n_alter_calls):
        dyn_grid.alter(prob_dict)

    grid_diff_counter = count_grid_differences(dyn_grid.grid.grid, dyn_grid_copy.grid.grid)

    assert grid_diff_counter == n_alter_calls


def test_alter_changes_goal(dyn_grid):
    prob_dict = {
    "alter_start_pos": 0,
    "alter_goal_pos": 1.0,
    "wall": 0,
    "lava": 0,
    "sand": 0,
    }
    start_pos = dyn_grid.agent_start_pos
    agent_pos = dyn_grid.agent_start_pos
    goal_pos = dyn_grid.goal_pos

    dyn_grid.alter(prob_dict)
    assert dyn_grid.agent_start_pos == start_pos
    assert dyn_grid.agent_pos == agent_pos

    assert dyn_grid.goal_pos != goal_pos


def test_solvability_check(dyn_grid, prob_dict):

    assert dyn_grid.alter(prob_dict)  # must be solvable after one alternation

    # create a grid with a wall around the agent
    dyn_grid_walled = DynamicMiniGrid()
    dyn_grid_walled.put_obj(Wall(), 1, 2)
    dyn_grid_walled.put_obj(Wall(), 2, 1)
    dyn_grid_walled.put_obj(Wall(), 2, 2)

    prob_dict = np.array([0, 1, 0, 0])  # move the goal, to ensure the agent remains locked

    assert not dyn_grid_walled.alter(prob_dict)







