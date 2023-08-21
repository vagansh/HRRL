import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

from hyperparam import Hyperparam


class Net_J(nn.Module):
    """ Net_J is the tentative of the agent to learn the world.
    It is an approximation of the real J, which is the expected 
    discounted drive over the lifetime of the agent
    (it also egals minus the expected value).
    The agents wants to minimize this discounted drive.
    """

    def __init__(self, hyperparam: Hyperparam):
        super(Net_J, self).__init__()
        n_neurons = hyperparam.cst_nets.n_neurons
        dropout_rate = hyperparam.cst_nets.dropout_rate
        shape_zeta = hyperparam.cst_agent.zeta_shape
        self.fc1 = nn.Linear(shape_zeta, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 1)

    def forward(self, x):
        """Return a real number. Not a vector"""
        x = self.fc1(x)
        x = sigmoid(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = sigmoid(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = sigmoid(x)
        return output


class Net_f(nn.Module):
    """ f is homogeneous to the rate of change of Zeta.
    For example, the rate of glucose consumption
    or the rate of consumption of water

    d_zeta = f((zeta, u)) * dt.
    Net_f is the tentative of the agent to model f.

    Zeta is the whole world seen by the agent.
    zeta = internal + external state
    """

    def __init__(self, hyperparam: Hyperparam) :
        super(Net_f, self).__init__()
        n_neurons = hyperparam.cst_nets.n_neurons
        dropout_rate = hyperparam.cst_nets.dropout_rate
        shape_zeta = hyperparam.cst_agent.zeta_shape
        n_actions = hyperparam.cst_actions.n_actions
        self.fc1 = nn.Linear(shape_zeta + n_actions, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, shape_zeta)

    def forward(self, x):
        """Return a speed homogeneous to zeta."""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        return output
