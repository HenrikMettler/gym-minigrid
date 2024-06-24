import numpy as np
import warnings

from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Lava, Sand, Wall


class DynamicMiniGrid(MiniGridEnv):
    """
    DynamicMiniGrid: Mini Grid Environment, that can dynamically change, by altering a single tile
    """

    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, agent_view_size=7, seed=1337):

        # Copied from EmptyEnv Todo: Make this class a child of EmptyEnv?
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__( grid_size=size, max_steps=4 * size * size,
                          see_through_walls=False, agent_view_size=agent_view_size, seed=seed)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner # Todo - should this change?
        self.goal_pos = (width-2, height-2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def alter(self, prob_dict, visibility_check=True):
        """
        Changes a single element of the environment. Automatically checks whether the environment
        can be (empirically in 5000 random steps) solved.
        Applies another change of the same alteration if environment is not solvable.

        :param prob_dict: dict. Dictionary of probabilties for each type of altering
            (change start or goal position, add wall/lava). Elements must sum to 1
        :param visibility_check: bool. If true, checks whether the agent can see the reward
            at the start and rejects such a solution.
        :return: None
        """

        if sum(prob_dict.values()) != 1.0:
            raise ValueError('Probabilities do not sum to 1')

        random_float = self.np_random.uniform()

        if random_float < prob_dict["alter_start_pos"]:
            self.alter_start_pos()

        elif random_float < prob_dict["alter_goal_pos"] + prob_dict["alter_start_pos"]:
            self.alter_goal_pos()

        elif random_float < prob_dict["wall"] + prob_dict["alter_goal_pos"] + prob_dict["alter_start_pos"]:
            self.set_or_remove_obj(Wall())

        elif random_float < prob_dict["lava"] + prob_dict["wall"] + prob_dict["alter_goal_pos"] + prob_dict["alter_start_pos"]:
            self.set_or_remove_obj(Lava())
        else:
            self.set_or_remove_obj(Sand())

    def is_solvable(self):
        # empirical check: let a random agent take max_steps and see if it visited the goal
        max_steps = 5000

        if self.height * self.width > 100:
            warnings.warn(f"Solvability takes {max_steps} with a random agent, "
                          f"thus might be wrong in large grids ", UserWarning)

        reachable_pos = [self.agent_start_pos]
        for _ in range(max_steps):
            # take a (random) step
            action = self.np_random.randint(low=0, high=3)  # 0 turn left, 1 turn right, 2 move
            self.step(action)
            reachable_pos.append(tuple(self.agent_pos))
            if self.grid.get(*self.agent_pos) is not None \
                    and isinstance(self.grid.get(*self.agent_pos), Lava):
                # if the agent walks on lava reset it to initial position and direction
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            if self.goal_pos in reachable_pos:
                # reset the agent to its starting position
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
                return True
        # reset the agent to its starting position
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        return False

    def alter_start_pos(self):

        def goal_in_view(pos, new_pos, dir, new_dir):
            self.agent_pos = new_pos
            self.agent_dir = new_dir
            return_value = self.in_view(*self.goal_pos)
            # reset the agent after check
            self.agent_pos = pos
            self.agent_dir = dir
            return return_value

        pos = self.agent_start_pos
        dir = self.agent_start_dir
        solvable = False
        n_tries_max = 10 * self.grid.width * self.grid.height
        n_tries = 0
        while (pos == self.agent_start_pos or not self.is_solvable()) and n_tries < n_tries_max:
            n_tries += 1
            new_pos = (self.np_random.randint(1, self.height - 1),  # 1, -1 to avoid boarders
                       self.np_random.randint(1, self.width - 1))
            new_dir = self.np_random.randint(0, 4)  # 4 possible directions
            if self.grid.get(
                    *new_pos) is not None or new_pos == pos:  # check field is empty and agent is not already there
                continue
            if self.visibility_check and goal_in_view(pos, new_pos, dir, new_dir):
                continue
            # set the new pos & dir if accepted
            self.agent_start_pos = new_pos
            self.agent_pos = new_pos
            self.agent_start_dir = new_dir
            self.agent_dir = new_dir

        if n_tries == n_tries_max:
            raise EnvironmentError(f"Could not alter the agent start position "
                          f"in a {n_tries_max} trials. Return without change")

    def alter_goal_pos(self):
        goal_pos = self.goal_pos
        n_tries_max = 10 * self.grid.width * self.grid.height
        n_tries = 0
        while (goal_pos == self.goal_pos or not self.is_solvable()) and n_tries < n_tries_max:
            new_goal_pos = (self.np_random.randint(1, self.height - 1),
                            self.np_random.randint(1, self.width - 1))
            if self.grid.get(*new_goal_pos) is not None or new_goal_pos == self.agent_start_pos:
                continue
            if self.visibility_check and self.in_view(*new_goal_pos):
                continue
            self.goal_pos = new_goal_pos  # change the attribute
            self.grid.set(*new_goal_pos, Goal())  # change the actual element in the grid
            self.grid.set(*goal_pos, None)  # remove the previous goal
            if not self.is_solvable():
                # revert the change
                self.goal_pos = goal_pos
                self.grid.set(*goal_pos, Goal())
                self.grid.set(*new_goal_pos, None)

        if n_tries == n_tries_max:
            raise Warning("Could not alter the agent start position "
                          "in a reasonable amount of trials. Return without change")

    def set_or_remove_obj(self, obj):

        solvable_and_changed = False

        while not solvable_and_changed:

            rand_pos = (self.np_random.randint(1, self.height-1),
                        self.np_random.randint(1, self.width-1))

            if rand_pos == self.agent_start_pos or rand_pos == self.goal_pos:
                continue

            # first condition needed because NoneType has no attribute type
            if self.grid.get(*rand_pos) is not None \
                    and self.grid.get(*rand_pos).type == obj.type:

                # remove obj
                self.grid.set(*rand_pos, None)
                solvable_and_changed = True # removing objects can not make it unsolvable

            else:  # replace even if there is an object of the other type
                current_obj = self.grid.get(*rand_pos)
                self.grid.set(*rand_pos, obj)
                solvable_and_changed = self.is_solvable()
                if not solvable_and_changed:
                    # revert the change before choosing another obj
                    self.grid.set(*rand_pos, current_obj)


    def respawn(self):
        """ alternative to the reset method (which initializes an empty grid at every timestep"""
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs