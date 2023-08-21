import csv

from hyperparam import Hyperparam
from environment import Environment
from agent import Agent, HomogeneousZeta
from actions import Actions
from nets import Net_J, Net_f
from plots import Plots

from typing import List

import numpy as np
import torch


class Algorithm:
    def __init__(self, hyperparam: Hyperparam, env: Environment, agent: Agent,
                 actions: Actions, net_J: Net_J, net_f: Net_f, plots: Plots):

        # CLASSES #########################################
        self.hp = hyperparam
        self.env = env
        self.agent = agent
        self.actions = actions
        self.net_J = net_J
        self.net_f = net_f
        self.plots = plots

        # UTILS ############################################
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.hp.cst_algo.learning_rate)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.hp.cst_algo.learning_rate)

        # TODO: ATTENTION AU DEEP COPY, si on met directement self.agent.zeta
        # au lieu de zeta_init (avec ou sans clone), ça ne marche pas
        zeta_init = HomogeneousZeta(self.hp)
        zeta_init.tensor = self.agent.zeta.tensor#.clone()
        self.historic_zeta = []
        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]

    def evaluate_action(self, action: int):
        """Return the score associated with the action.

        In this function, we do not seek to update the Net_F and Net_J,
         so we use the eval mode.
        But we still seek to use the derivative of the Net_F according to zeta.
         So we use require_grad = True.
        Generally, only the parameters of a neural network are on 
        require_grad = True.
        But here we must use zeta.require_grad = True.

        Parameters:
        ----------
        action : int

        Returns:
        --------
        the score : pytorch float.
        """
        _zeta_tensor = self.agent.zeta.tensor
        # f is a neural network taking one vector.
        # But this vector contains the information of zeta and u.
        # The u is the one-hot-encoded control associated with the action a
        zeta_u = torch.cat(
            [_zeta_tensor, torch.zeros(self.hp.cst_actions.n_actions)])
        index_control = len(_zeta_tensor) + action
        zeta_u[index_control] = 1

        # Those lines are only used to accelerate the computations but are not
        # strickly necessary.
        # because we don't want to compute the gradient wrt theta_f and theta_J.
        for param in self.net_f.parameters():
            param.requires_grad = False
        for param in self.net_J.parameters():
            param.requires_grad = False

        # In the Hamilton Jacobi Bellman equation, we derivate J by zeta.
        # But we do not want to propagate this gradient.
        # We seek to compute the gradient of J with respect to zeta_to_J.
        _zeta_tensor.requires_grad = True
        # zeta_u_to_f.require_grad = False : This is already the default.

        # Deactivate dropout and batchnorm but continues to accumulate the gradient.
        # This is the reason it is generally used paired with "with torch.no_grad()"
        self.net_J.eval()
        self.net_f.eval()

        # in the no_grad context, all the results of the computations will have
        # requires_grad=False,
        # even if the inputs have requires_grad=True
        # If you want to freeze part of your model and train the rest, you can set
        # requires_grad of the parameters you want to freeze to False.
        f = self.net_f.forward(zeta_u).detach()

        new_zeta_tensor = _zeta_tensor + self.hp.cst_algo.time_step * f
        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.tensor = new_zeta_tensor

        instant_reward = self.agent.drive(new_zeta)


        grad_ = torch.autograd.grad(
            self.net_J(_zeta_tensor), _zeta_tensor)[0]
        future_reward = torch.dot(grad_, self.net_f.forward(zeta_u))
        future_reward = future_reward.detach()

        score = instant_reward + future_reward

        _zeta_tensor.requires_grad = False
        # BIZARRE
        self.agent.zeta.tensor = _zeta_tensor

        for param in self.net_f.parameters():
            param.requires_grad = True
        for param in self.net_J.parameters():
            param.requires_grad = True

        self.net_J.train()
        self.net_f.train()
        return instant_reward, future_reward

    def simulation_one_step(self, k: int):
        """Simulate one step.
        """
        _zeta = self.agent.zeta.tensor
        # if you are exacly on 0 (empty resource) you get stuck
        # because of the nature of the differential equation.
        # It is commented because it is done below
        #for i in range(self.agent.zeta.n_homeostatic):
        #    # zeta = x - x_star
        #    if _zeta[i] + self.hp.cst_agent.x_star[i] < self.hp.cst_agent.min_resource:
        #        _zeta[i] = -self.hp.cst_agent.x_star[i] + self.hp.cst_agent.min_resource

        possible_actions = [cstr(self.agent, self.env) for cstr in self.actions.df.loc[:, "constraints"].tolist()]
        indexes_possible_actions = [i for i in range(
            self.hp.cst_actions.n_actions) if possible_actions[i]]

        # The default action is doing nothing. Like people in real life.
        # There is a [0] because this is a list of one element.
        index_default_action = self.actions.df.loc[:, "name"] == self.hp.cst_actions.default_action
        action = self.actions.df.index[index_default_action][0]

        exploit_explore = ' '
        i_reward = 0
        f_reward = 0

        if np.random.random() <= self.hp.cst_algo.eps:
            action = np.random.choice(indexes_possible_actions)
            exploit_explore = "explore"
            print("INSIDE EXPLORE ")
            i_reward, f_reward = self.evaluate_action(action)
            discounted_reward = i_reward + f_reward

        else:
            # TODO: use batch
            exploit_explore = "exploit"
            best_score = np.Inf
            for act in indexes_possible_actions:
                i_reward, f_reward = self.evaluate_action(act)
                score = i_reward + f_reward
                if score < best_score:
                    best_score = score
                    action = act

        zeta_u = torch.cat(
            [_zeta, torch.zeros(self.hp.cst_actions.n_actions)])
        zeta_u[len(_zeta) + action] = 1

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.tensor = self.actions.df.loc[action, "new_state"](self.agent, self.env).tensor
        _new_zeta = new_zeta.tensor

        self.historic_zeta.append(new_zeta)
        print(len(self.historic_zeta))

        predicted_new_zeta = _zeta + self.hp.cst_algo.time_step *  self.net_f.forward(zeta_u)

        coeff = self.actions.df.loc[action, "coefficient_loss"]

        Loss_f = coeff * torch.dot(_new_zeta - predicted_new_zeta,
                                   _new_zeta - predicted_new_zeta)

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_f.backward()
        self.optimizer_J.zero_grad()
        self.optimizer_f.step()

        _zeta.requires_grad = True

        # if drive = d(\zeta_t)= 1 and globally convex environment (instant
        # and long-term improvements are in the same direction)

        # futur drive = d(\zeta_t, u_a) = 0.9
        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.tensor = _new_zeta
        instant_drive = self.agent.drive(new_zeta)

        # negative
        delta_deviation = torch.dot(torch.autograd.grad(self.net_J(_zeta),
                                                        _zeta)[0],
                                    self.net_f.forward(zeta_u))

        # 0.1 current deviation
        discounted_deviation = - torch.log(torch.tensor(self.hp.cst_algo.gamma)) * \
            self.net_J.forward(_zeta)
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        _zeta.requires_grad = False

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_J.backward()
        self.optimizer_f.zero_grad()
        self.optimizer_J.step()

        self.agent.zeta.tensor = _new_zeta
        #print(_new_zeta)



        for index in self.hp.cst_agent.features_to_index["homeostatic"]:
            #print(self.agent.zeta.tensor[index], self.agent.x_star.tensor[index] , self.hp.cst_agent.min_resource)
            if self.agent.zeta.tensor[index] + self.agent.x_star.tensor[index] < self.hp.cst_agent.min_resource:
                print("I AM CALLED")
                self.agent.zeta.tensor[index] = -self.agent.x_star.tensor[index] + self.hp.cst_agent.min_resource

        loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()[0]])

        # save historic
        #self.historic_zeta.append(self.agent.zeta)
        self.historic_actions.append(action)
        self.historic_losses.append(loss)

        action_string = [str(_zeta.detach().numpy()[0]), str(_zeta.detach().numpy()[1]), str(_zeta.detach().numpy()[2])
            , str(_zeta.detach().numpy()[3]), str(_zeta.detach().numpy()[4]), str(_zeta.detach().numpy()[5]),
                         exploit_explore,

                         str(action), self.actions.df.loc[action, "name"],
                         str(i_reward), str(f_reward), str((i_reward + f_reward)),

                         str(new_zeta.tensor.detach().numpy()[0]),
                         str(new_zeta.tensor.detach().numpy()[1]),
                         str(new_zeta.tensor.detach().numpy()[2]),
                         str(new_zeta.tensor.detach().numpy()[3]),
                         str(new_zeta.tensor.detach().numpy()[4]),
                         str(new_zeta.tensor.detach().numpy()[5]),

                         str(instant_drive), str(discounted_deviation.detach().numpy()[0]),

                         str(predicted_new_zeta.detach().numpy()[0]),
                         str(predicted_new_zeta.detach().numpy()[1]),
                         str(predicted_new_zeta.detach().numpy()[2]),
                         str(predicted_new_zeta.detach().numpy()[3]),
                         str(predicted_new_zeta.detach().numpy()[4]),
                         str(predicted_new_zeta.detach().numpy()[5]),
                         str(Loss_J.detach().numpy()[0]), str(Loss_f.detach().numpy()),

                         ]

        return action_string

    def simulation(self):


        with open("easy.csv", "w", newline='') as output:
            writer = csv.writer(output)
            writer.writerow(
                ["C_R1", "C_R2", "C_Muscular_Fatigue", "C_Sleep_Fatigue", "C_X", "C_Y", "Exploit_Explore", "A_num",
                 "Act",
                 "Instant_reward", "Future_reward", "Discounted_sum_of_rewards",
                 "NS_r1", "NS_r2", "NS_mf", "NS_sf", "NS_x", "NS_y", "Instant_Drive", "Discounted_deviation",
                 "PNS_r1", "PNS_r2", "PNS_mf", "PNS_sf", "PNS_x", "PNS_y", "Loss_J", "Loss_f"])

            for k in range(self.hp.cst_algo.N_iter):
                action_string = self.simulation_one_step(k)
                writer.writerow(action_string)

                if ((k % self.hp.cst_algo.N_print) == 0) or (k == self.hp.cst_algo.N_iter-1):
                    print("Iteration:", k, "/", self.hp.cst_algo.N_iter - 1)
                    #print("Zeta before action:", self.historic_zeta[-2].tensor)
                    action = self.historic_actions[-1]
                    print("Action:", action, self.actions.df.loc[action, "name"])
                    print("")

                if (k % self.hp.cst_algo.cycle_plot == 0) or (k == self.hp.cst_algo.N_iter-1):
                    self.plots.plot(frame=k,
                                    env=self.env,
                                    historic_zeta=self.historic_zeta,
                                    historic_actions=self.historic_actions,
                                    actions=self.actions,
                                    historic_losses=self.historic_losses,
                                    net_J=self.net_J)

                if ((k % self.hp.cst_algo.N_save_weights) == 0) or (k == self.hp.cst_algo.N_iter-1):
                    torch.save(self.net_J.state_dict(), 'weights/weights_net_J')
                    torch.save(self.net_f.state_dict(), 'weights/weights_net_f')

