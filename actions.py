from typing import Any, Dict

from environment import Environment
from agent import HomogeneousZeta, Agent
from hyperparam import Hyperparam

import torch
import numpy as np

from typing import Callable

import pandas as pd


class Actions:
    """Defining the possible actions for the agent
    in its environment.
    """

    def __init__(self, hyperparam: Hyperparam) -> None:

        self.hp = hyperparam

        actions_list = []

        # USEFUL FOR ACTIONS OF WALKING

        def new_state_and_constraints_walking(direction: str):
            if direction not in ['right', 'left', 'up', 'down']:
                raise ValueError('direction should be right, left, up or down.')
            elif direction == 'right':
                control_walking_right = HomogeneousZeta(self.hp)
                control_walking_right.muscular_fatigue = self.hp.cst_actions.fatigue_walking
                control_walking_right.x = self.hp.cst_agent.walking_speed
                control_walking = control_walking_right
            elif direction == 'left':
                control_walking_left = HomogeneousZeta(self.hp)
                control_walking_left.muscular_fatigue = self.hp.cst_actions.fatigue_walking
                control_walking_left.x = -self.hp.cst_agent.walking_speed
                control_walking = control_walking_left
            elif direction == 'up':
                control_walking_up = HomogeneousZeta(self.hp)
                control_walking_up.muscular_fatigue = self.hp.cst_actions.fatigue_walking
                control_walking_up.y = self.hp.cst_agent.walking_speed
                control_walking = control_walking_up
            elif direction == 'down':
                control_walking_down = HomogeneousZeta(self.hp)
                control_walking_down.muscular_fatigue = self.hp.cst_actions.fatigue_walking
                control_walking_down.y = -self.hp.cst_agent.walking_speed
                control_walking = control_walking_down

            def new_state_walking(agent: Agent, env: Environment) -> HomogeneousZeta:
                new_zeta = HomogeneousZeta(self.hp)
                new_zeta.tensor = agent.euler_method(agent.zeta, control_walking).tensor
                return new_zeta

            def constraints_walking(agent: Agent, env: Environment) -> bool:
                new_zeta = new_state_walking(agent, env)
                return ((agent.zeta.sleep_fatigue < self.hp.cst_agent.max_sleep_fatigue)
                        and (agent.zeta.muscular_fatigue < self.hp.cst_agent.max_muscular_fatigue)
                        and env.is_point_inside(new_zeta.x, new_zeta.y))

            return new_state_walking, constraints_walking

        # ACTIONS OF WALKING RIGHT, LEFT, UP AND DOWN

        for direction in ['right', 'left', 'up', 'down']:
            new_state_walking, constraints_walking = new_state_and_constraints_walking(direction)
            action_walking = {
                "name": f"walking_{direction}",
                "definition": f"Walking one step {direction}.",
                "new_state": new_state_walking,
                "constraints": constraints_walking,
                "coefficient_loss": self.hp.cst_actions.coefficient_loss_small_action,
            }
            actions_list.append(action_walking)

        # ACTION OF SLEEPING

        def new_state_sleeping(agent: Agent, env: Environment) -> HomogeneousZeta:
            control_sleeping = HomogeneousZeta(self.hp)
            control_sleeping.sleep_fatigue = self.hp.cst_actions.recover_sleeping
            duration_sleep = self.hp.cst_agent.n_min_time_sleep * self.hp.cst_algo.time_step
            new_zeta = HomogeneousZeta(self.hp)
            new_zeta.tensor = agent.integrate_multiple_steps(
                duration_sleep, agent.zeta, control_sleeping).tensor
            return new_zeta

        def constraints_sleeping(agent: Agent, env: Environment) -> bool:
            return (agent.zeta.sleep_fatigue > self.hp.cst_agent.min_sleep_fatigue)

        action_sleeping = {
            "name": "sleeping",
            "definition": "Sleeping for a fixed time period to recover from muscular and sleep fatigues.",
            "new_state": new_state_sleeping,
            "constraints": constraints_sleeping,
            "coefficient_loss": self.hp.cst_actions.coefficient_loss_big_action,
        }
        actions_list.append(action_sleeping)

        # ACTION OF DOING NOTHING

        def new_state_doing_nothing(agent: Agent, env: Environment) -> HomogeneousZeta:
            control_doing_nothing = HomogeneousZeta(self.hp)
            new_zeta = HomogeneousZeta(self.hp)
            new_zeta.tensor = agent.euler_method(agent.zeta, control_doing_nothing).tensor
            return new_zeta

        def constraints_doing_nothing(agent: Agent, env: Environment) -> bool:
            return (agent.zeta.sleep_fatigue < self.hp.cst_agent.max_sleep_fatigue)
        #sometimes it is better to do nothing, than to move around


        action_doing_nothing = {
            "name": "doing_nothing",
            "definition": "Standing still and doing nothing.",
            "new_state": new_state_doing_nothing,
            "constraints": constraints_doing_nothing,
            "coefficient_loss": self.hp.cst_actions.coefficient_loss_small_action,
        }
        actions_list.append(action_doing_nothing)

        # USEFUL FOR ACTIONS OF CONSUMING A RESOURCE

        def new_state_and_constraints_consuming_resource(res: int):
            if (res < 0) or (res >= self.hp.difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_consuming_resource(agent: Agent, env: Environment) -> HomogeneousZeta:
                    control_consuming_resource = HomogeneousZeta(self.hp)
                    control_consuming_resource.set_resource(res, self.hp.cst_actions.consumption_resource)
                    new_zeta = HomogeneousZeta(self.hp)
                    new_zeta.tensor = agent.euler_method(agent.zeta, control_consuming_resource).tensor
                    return new_zeta

                def constraints_consuming_resource(agent: Agent, env: Environment) -> bool:
                    return ((agent.zeta.sleep_fatigue < self.hp.cst_agent.max_sleep_fatigue)
                            and env.is_near_resource(agent.zeta.x, agent.zeta.y, res)
                            and (agent.zeta.get_resource(res) < self.hp.cst_agent.max_eating))

                return new_state_consuming_resource, constraints_consuming_resource

        # ACTIONS OF CONSUMING A RESOURCE

        for res in range(self.hp.difficulty.n_resources):
            new_state_consuming_resource, constraints_consuming_resource = new_state_and_constraints_consuming_resource(
                res)
            action_consuming_resource = {
                "name": f"consuming_resource_{res}",
                "definition": f"Consuming resource {res}.",
                "new_state": new_state_consuming_resource,
                "constraints": constraints_consuming_resource,
                "coefficient_loss": self.hp.cst_actions.coefficient_loss_small_action,
            }
            actions_list.append(action_consuming_resource)

        # USEFUL FOR ACTIONS OF GOING TO A RESOURCE

        def new_state_and_constraints_going_to_resource(res: int):
            if (res < 0) or (res >= self.hp.difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_going_to_resource(agent: Agent, env: Environment) -> HomogeneousZeta:
                    control_going_to_resource = HomogeneousZeta(self.hp)
                    control_going_to_resource.muscular_fatigue = self.hp.cst_actions.fatigue_walking
                    dist = env.distance_to_resource(agent.zeta.x, agent.zeta.y, res)
                    duration_walking = (dist / self.hp.cst_agent.walking_speed) * self.hp.cst_algo.time_step

                    new_zeta = HomogeneousZeta(self.hp)
                    new_zeta.tensor = agent.integrate_multiple_steps(
                        duration_walking, agent.zeta, control_going_to_resource).tensor
                    new_zeta.x = self.hp.cst_env.resources[res].x
                    new_zeta.y = self.hp.cst_env.resources[res].y

                    return new_zeta

                def constraints_going_to_resource(agent: Agent, env: Environment) -> bool:
                    new_zeta = new_state_going_to_resource(agent, env)
                    return ((new_zeta.sleep_fatigue < self.hp.cst_agent.max_sleep_fatigue)
                            and (new_zeta.muscular_fatigue < self.hp.cst_agent.max_muscular_fatigue)
                            and env.is_resource_visible(agent.zeta.x, agent.zeta.y, res)
                            and (env.distance_to_resource(agent.zeta.x, agent.zeta.y, res) > 0))

                return new_state_going_to_resource, constraints_going_to_resource

        # ACTIONS OF GOING TO A RESOURCE

        for res in range(self.hp.difficulty.n_resources):
            new_state_going_to_resource, constraints_going_to_resource = new_state_and_constraints_going_to_resource(
                res)
            action_going_to_resource = {
                "name": f"going_to_resource_{res}",
                "definition": f"Going to resource {res}.",
                "new_state": new_state_going_to_resource,
                "constraints": constraints_going_to_resource,
                "coefficient_loss": self.hp.cst_actions.coefficient_loss_big_action,
            }
            actions_list.append(action_going_to_resource)

        # CREATION OF THE DATAFRAME

        self.df = pd.DataFrame(actions_list)
