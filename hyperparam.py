import numpy as np
import torch

from typing import Literal, List

from dataclasses import dataclass


class Difficulty:
    def __init__(self, level: Literal["EASY", "MEDIUM"]):
        if level == "EASY":
            self.n_resources: Literal[2, 4] = 2
            self.type_env: Literal["polygon", "square"] = "square"

        elif level == "MEDIUM":
            self.n_resources: Literal[2, 4] = 4
            self.type_env: Literal["polygon", "square"] = "polygon"


@dataclass
class Point:
    """Point delimiting the boundaries of the environment."""
    x: float
    y: float

@dataclass
class ResourceDC:
    """Resource representing a type of resource in the environment."""
    x: float
    y: float
    r: float
    color: str

class Cst_env:
    def __init__(self, difficulty: Difficulty):
        coord_env_polygon = [Point(0, 1),
                             Point(0, 5),
                             Point(1, 5),
                             Point(1, 3),
                             Point(2, 3),
                             Point(2, 4),
                             Point(3, 4),
                             Point(3, 6),
                             Point(8, 6),
                             Point(8, 5),
                             Point(5, 5),
                             Point(5, 3),
                             Point(6, 3),
                             Point(6, 0),
                             Point(5, 0),
                             Point(5, 2),
                             Point(4, 2),
                             Point(4, 1)]

        coord_env_square = [Point(0, 0),
                            Point(10, 0),
                            Point(10, 10),
                            Point(0, 10),]
        
        self.coord_env = coord_env_polygon if difficulty.type_env == "polygon" else coord_env_square
        
        four_resources: List[ResourceDC] = [
            ResourceDC(x=0.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=3.5, y=1.5, r=0.3, color='blue'),
            ResourceDC(x=7, y=5.5, r=0.3, color='orange'),
            ResourceDC(x=5.5, y=0.75, r=0.3, color='green'),
        ]

        two_resources: List[ResourceDC] = [
            ResourceDC(x=0.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=3.5, y=1.5, r=0.3, color='blue'),
        ]
        
        self.resources = two_resources if difficulty.n_resources == 2 else four_resources
        
        coord_env_x = [point.x for point in self.coord_env]
        coord_env_y = [point.y for point in self.coord_env]

        self.width = np.max(coord_env_x) - np.min(coord_env_x)
        self.height = np.max(coord_env_y) - np.min(coord_env_y)


class Cst_agent:
    def __init__(self, difficulty: Difficulty):
        self.default_pos_x = 2
        self.default_pos_y = 2

        features_to_index = {}
        for i in range(difficulty.n_resources):
            features_to_index[f"resource_{i}"] = i
        features_to_index["muscular_fatigue"] = difficulty.n_resources
        features_to_index["sleep_fatigue"] = difficulty.n_resources + 1
        features_to_index["x"] = difficulty.n_resources + 2
        features_to_index["y"] = difficulty.n_resources + 3
        features_to_index["homeostatic"] = [i for i in range(difficulty.n_resources + 2)]
        features_to_index["non_homeostatic"] = [i for i in range(difficulty.n_resources + 2,
                                                                 difficulty.n_resources + 4)]
        self.features_to_index = features_to_index

        self.zeta_shape = difficulty.n_resources + 4 # muscular, aware, x, y

        # Homeostatic setpoint
        val_x_star = {}
        for i in range(difficulty.n_resources):
            val_x_star[f"resource_{i}"] = i + 1
        val_x_star["muscular_fatigue"] = 0
        val_x_star["sleep_fatigue"] = 0
        self.val_x_star = val_x_star

        # Coefficients of the function f related to the
        # differential equation
        val_coeff_eq_diff = {}
        for i in range(difficulty.n_resources):
            val_coeff_eq_diff[f"resource_{i}"] = -0.05
        val_coeff_eq_diff["muscular_fatigue"] = -0.008
        val_coeff_eq_diff["sleep_fatigue"] = 0.0005
        self.val_coeff_eq_diff = val_coeff_eq_diff

        self.walking_speed = 0.1

        # Minimum duration when the agent sleeps
        self.n_min_time_sleep = 1000

        # If the agent is too tired (sleep), it automatically sleeps.
        self.max_sleep_fatigue = 10
        # If the agent is not tired (sleep) enough, it cannot sleep
        self.min_sleep_fatigue = 1
        # If the agent is too tired (muscular), it cannot move
        self.max_muscular_fatigue = 6

        self.max_eating = 8

        # If any of the agent's resources are less than min_resource,
        # we raise it to min_resource.
        # Because the dynamics is controlled by an exponential function,
        # if one resource equals 0, it can never be raised again.
        self.min_resource = 0.1


class Cst_actions:
    def __init__(self, difficulty: Difficulty, cst_agent: Cst_agent):
        self.n_actions = 6 + 2 * difficulty.n_resources

        self.fatigue_walking = 0.01

        self.recover_sleeping = -0.001

        self.consumption_resource = 0.1

        self.default_action = "doing_nothing"

        self.coefficient_loss_small_action = 1
        self.coefficient_loss_big_action = 100


class Cst_nets:
    def __init__(self):
        self.n_neurons = 128
        self.dropout_rate = 0.15


class Cst_algo:
    def __init__(self):
        # Simulation
        self.time_step = 1
        self.N_iter = 100

        # RL learning
        self.eps = 0.3  # random actions
        self.gamma = 0.99  # discounted rate
        self.tau = 0.001  # not used yet (linked with the target function)

        # Gradient descent
        self.learning_rate = 0.001

        # Verbose
        self.N_print = 1
        self.cycle_plot = self.N_iter - 1
        self.N_rolling = 5

        # Save neural networks weights every N_save_weights step
        self.N_save_weights = 1000


class Cst_tests:
    def __init__(self, difficulty: Difficulty):
        if difficulty.type_env == "square":
            self.is_point_inside = [
                (-1, 1, False),
                (1, 1, True),
                (9, 1, True),
                (11, 1, False),
            ]
            self.is_segment_inside = [
                # xa, ya, xb, yb,
                (3, 4, 5, 6, True),
                (1, 1, 9, 9, True),
                (1, 1, 1, 11, False),
                (3, 4, 12, 6, False),
            ]

        elif difficulty.type_env == "polygon":
            self.is_point_inside = [
                (0.1, 1.1, True), # up right A
                (-0.1, 0.9, False), # down left A
                (-0.1, 5, False), # left B
                (0.1, 4.9, True), # right down B
            ]
            self.is_segment_inside = [
                # xa, ya, xb, yb,
                (0.1, 1.1, -0.1, 0.9, False), # up right A - down left A
                (-0.1, 0.9, 0.1, 1.1, False), # down left A - up right A
                (-0.1, 0.9, 0.1, 4.9, False), # down left A - right down B
                (0.1, 1.1, 0.1, 4.9, True), # up right A - right down B
                (0.1, 1.1, 0.1, 5.1, False), # up right A - right up B
                (0.5, 1.5, 0.5, 4.5, True), # other test
                (0.5, 4.5, 0.5, 1.5, True), # other test
                (0.5, 6.5, 0.5, 1.5, False), # other test
            ]

        self.visualization_scale = 10


class Hyperparam:
    def __init__(self, level: Literal["EASY", "MEDIUM"]):
        self.difficulty = Difficulty(level)
        self.cst_env = Cst_env(self.difficulty)
        self.cst_agent = Cst_agent(self.difficulty)
        self.cst_actions = Cst_actions(self.difficulty, self.cst_agent)
        self.cst_nets = Cst_nets()
        self.cst_algo = Cst_algo()
        self.cst_tests = Cst_tests(self.difficulty)

