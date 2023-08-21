import unittest

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import set_all_seeds
from hyperparam import Hyperparam
from environment import Environment
from agent import Agent
from actions import Actions
from nets import Net_J, Net_f
from plots import Plots
from algorithm import Algorithm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


level = input("Enter the level between EASY or MEDIUM:")
if level not in ["EASY", "MEDIUM"]:
    raise ValueError("level should be EASY or MEDIUM.")


class TestEnvironnement(unittest.TestCase):
    hp = Hyperparam(level)
    env = Environment(hp)

    def test_is_point_inside(self):
        points = self.hp.cst_tests.is_point_inside
        for (x, y, inside) in points:
            self.assertEqual(self.env.is_point_inside(x, y), inside)

    def test_is_segment_inside(self):
        segments = self.hp.cst_tests.is_segment_inside
        for (xa, ya, xb, yb, inside) in segments:
            self.assertEqual(self.env.is_segment_inside(xa, xb, ya, yb), inside)

    def test_visualization_env(self):
        scale = self.hp.cst_tests.visualization_scale
        width = self.hp.cst_env.width * scale
        height = self.hp.cst_env.height * scale

        values = np.zeros((width, height))
        for i in range(width): # x
            for j in range(height): # y
                values[i, j] = 1*self.env.is_point_inside(i/scale, j/scale)

        plt.imshow(values.T, cmap='cool', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()

        # Check visually if the environment is correct


class TestHRL(unittest.TestCase):
    def test_simulation(self):
        seed = 0
        set_all_seeds(seed)

        hyperparam = Hyperparam(level)
        hyperparam.cst_algo.N_print = hyperparam.cst_algo.N_iter - 1

        env = Environment(hyperparam)
        agent = Agent(hyperparam)
        actions = Actions(hyperparam)
        net_J = Net_J(hyperparam)
        net_f = Net_f(hyperparam)
        plots = Plots(hyperparam)
        algo = Algorithm(hyperparam, env, agent, actions, net_J, net_f, plots)
        algo.simulation()

        # Check manually the values of zeta and the
        # dashboard plot at the end of the simulation


if __name__ == '__main__':
    unittest.main()


