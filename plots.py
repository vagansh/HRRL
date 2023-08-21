from matplotlib import lines
import matplotlib.lines as lines
from hyperparam import Hyperparam
from environment import Environment
from agent import HomogeneousZeta, Agent
from actions import Actions
from nets import Net_J

from typing import List

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


class Plots:
    """Plots displayed in the simulation.
    """
    def __init__(self, hyperparam: Hyperparam) -> None:
        self.hp = hyperparam

    def plot_resources(self, ax, frame: int, historic_zeta: List[HomogeneousZeta]):
        """Plot the resource historic as a function of time.
        x-axis is step and not time.
        """
        zeta_shape = self.hp.cst_agent.zeta_shape
        zeta_meaning = list(self.hp.cst_agent.features_to_index.keys())[:zeta_shape]

        historic_zeta_tensor = [zeta.tensor.detach(
        ).numpy() for zeta in historic_zeta[:frame+1]]

        df = pd.DataFrame(historic_zeta_tensor, columns=zeta_meaning)
        df.plot(ax=ax, grid=True, yticks=list(range(0, 10)))
        ax.set_ylabel('value')
        ax.set_xlabel('frames')
        ax.set_title("Evolution of the resource")

    def plot_position(self, ax, env: Environment, zeta: HomogeneousZeta, historic_zeta):
        """Plot the position.
        x-axis is step and not time.
        """
        env.plot(ax=ax)  # initialisation of plt with background
        x = zeta.x
        y = zeta.y

        color = "grey"
        patch_circle = Circle((x, y), 0.2, color=color)
        # ax.add_patch(patch_circle)
        # ax.text(x, y, "agent")
        # print("Line plot data",  historic_zeta[-2].tensor,historic_zeta[-1].tensor)
        print("length of historic zeta is ", len(historic_zeta))

        for i in range(len(historic_zeta)):
            if (i+1) < len(historic_zeta):
                line = lines.Line2D([ historic_zeta[i].x, historic_zeta[i+1].x],
                                    [ historic_zeta[i].y, historic_zeta[i+1].y],
                                    lw=2, color='black',
                                    axes=ax, alpha=0.7)
                ax.add_line(line)
        ax.add_patch(patch_circle)
        ax.text(x, y, "agent")

        ax.set_title("Position of the agent.")

    def plot_loss(self, ax, frame: int, historic_losses: List):
        """Plot the loss in order to control the learning of the agent.
        x-axis is step and not time.
        """
        loss_meaning = [
            "Loss of the transition function $L_f$",
            "Loss of the deviation function $L_J$",
        ]

        df = pd.DataFrame(historic_losses[:frame+1],
                          columns=loss_meaning)
        df = df.rolling(window=self.hp.cst_algo.N_rolling).mean()
        df.plot(ax=ax, grid=True, logy=True)
        ax.set_ylabel('value of the losses')
        ax.set_xlabel('frames')
        ax.set_title(
            f"Evolution of the log-loss (moving average with "
            f"{self.hp.cst_algo.N_rolling} frames)")

    def compute_mask(self, scale: int, env: Environment):
        """Compute the mask indicating if a discretized value is inside
        or outside the environment.
        """
        n_X = self.hp.cst_env.width*scale
        n_Y = self.hp.cst_env.height*scale
        is_inside = np.zeros((n_X, n_Y))
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                is_inside[i, j] = env.is_point_inside(i/scale, j/scale)
        return is_inside

    NpArray = type(np.array([]))

    def plot_J(self, ax, fig, resource_id: int, scale: int, is_inside: NpArray,
               net_J: Net_J, env: Environment):
        """Plot of the learned J function.
        scale: number of plotted points for a unt square.
        is_inside: np.ndarray.
        """
        net_J.eval()

        n_X = self.hp.cst_env.width * scale
        n_Y = self.hp.cst_env.height * scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        # We could optimize this plot by using a batch with each element of
        # the batch representing one pixel in the image.
        # But this function represents only 1/8 of the total execution time.
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                if is_inside[i, j]:  # TODO : use torch batch
                    # We are at the optimum for three out of the 4 resources
                    # but one resources varies alongside with the coordinates.
                    # No muscular nor sleep fatigues.
                    zeta = HomogeneousZeta(self.hp)
                    zeta.x = i/scale
                    zeta.y = j/scale
                    zeta.set_resource(resource_id, -self.hp.cst_agent.val_x_star[f"resource_{resource_id}"])

                    with torch.no_grad():
                        values[i, j] = net_J(zeta.tensor).detach().numpy()

        im = ax.imshow(X=values.T, cmap="YlGnBu", norm=Normalize())
        ax.axis('off')
        ax.invert_yaxis()

        env.plot_resources(ax, scale, resources=[resource_id])

        ax.set_title(f'Deviation function (resource {resource_id} missing)')
        cbar = fig.colorbar(im, extend='both', shrink=0.4, ax=ax)

    def plot(self, frame: int, env: Environment, historic_zeta: List[HomogeneousZeta],
             historic_actions: List, actions: Actions, historic_losses: List,
             net_J: Net_J, scale=5):
        """Plot the position and the ressources of the agent.
        """
        fig = plt.figure(figsize=(16, 16))
        last_action = historic_actions[frame]
        fig.suptitle(
            (f'Dashboard. Frame: {frame} - last action: '
                f'{last_action}: {actions.df.loc[last_action, "name"]} '),
            fontsize=16)
        shape = (4, 4)

        ax_resource = plt.subplot2grid(shape, (0, 0), colspan=4)
        self.plot_resources(ax=ax_resource, frame=frame, historic_zeta=historic_zeta)

        zeta = historic_zeta[frame]
        ax_env = plt.subplot2grid(shape, (1, 0), colspan=2, rowspan=2)
        if frame >1:
            print("frame is ", frame)
            self.plot_position(ax=ax_env, env=env, zeta=zeta, historic_zeta = historic_zeta)

        ax_loss = plt.subplot2grid(shape, (1, 2), colspan=2, rowspan=2)
        self.plot_loss(ax=ax_loss, frame=frame, historic_losses=historic_losses)

        is_inside = self.compute_mask(scale=scale, env=env)
        for resource in range(self.hp.difficulty.n_resources):
            axs_J = plt.subplot2grid(shape, (3, resource))
            self.plot_J(ax=axs_J,
                        fig=fig, resource_id=resource,
                        scale=scale, is_inside=is_inside,
                        net_J=net_J, env=env)

        plt.tight_layout()
        name_fig = f"images/frame_{frame}"
        plt.savefig(name_fig)
        #print(name_fig)
        plt.close(fig)

