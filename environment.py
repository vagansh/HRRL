from hyperparam import Hyperparam, Point

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Environment:
    def __init__(self, hyperparam : Hyperparam):
        self.hp = hyperparam

    def is_point_inside(self, x: float, y: float) -> bool:
        """Check if a point (x,y) is inside the polygon.
        To do this, we look at the number of sides of the polygon at the left
        of the point.
        """
        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.hp.cst_env.coord_env + [self.hp.cst_env.coord_env[0]]
        n_left = 0

        def is_left(x0, y0, y1):
            cstr_1_y = (y0 > y) and (y1 <= y)
            cstr_2_y = (y0 <= y) and (y1 > y)
            cstr_x = (x0 <= x)
            if (cstr_1_y or cstr_2_y) and cstr_x:
                return True
            return False

        for i, point in enumerate(coords[:-1]):
            if is_left(point.x, point.y, coords[i + 1].y):
                n_left += 1
        if n_left % 2 == 1:
            return True
        else:
            return False

    def is_segment_inside(self, xa: float, xb: float, ya: float, yb: float) -> bool:
        """Check if the segment AB with A(xa, ya) and B(xb, yb) is completely
        inside the polygon.
        To do this, we look at the number of intersections between the segment
        and the sides of the polygon.
        """
        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.hp.cst_env.coord_env + [self.hp.cst_env.coord_env[0]]

        def point_in_seg(point: Point, A: Point, B: Point):
            """Check if a point on the line AB is actually
            on this segment (or just aligned to it)."""
            in_seg = (point.x >= min(A.x, B.x)) and \
                     (point.x <= max(A.x, B.x)) and \
                     (point.y >= min(A.y, B.y)) and \
                     (point.y <= max(A.y, B.y))
            return in_seg

        def is_inter(inter: Point, border0: Point, border1: Point):
            """Check if the intersection between the segment [A, B] 
            and the border number i is both inside [A, B] and the border."""
            inter_in_AB = point_in_seg(inter, Point(x=xa, y=ya), Point(x=xb, y=yb))
            if not inter_in_AB:
                return False
            inter_in_border = point_in_seg(inter, border0, border1)
            if not inter_in_border:
                return False
            return True

        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i, point in enumerate(coords[:-1]):
                if point.x == coords[i + 1].x:
                    inter = Point(x=point.x, y=alpha_1 * point.x + beta_1)
                    if is_inter(inter, point, coords[i + 1]):
                        return False
                else:
                    if ya == yb:
                        if ya == point.y:
                            inter_in_border = (min(xa, xb) <=
                                               max(point.x, coords[i + 1].x)) and \
                                              (max(xa, xb) >=
                                               min(point.x, coords[i + 1].x))
                            if inter_in_border:
                                return False
                    else:
                        inter = Point(x=(point.y - beta_1) / alpha_1, y=point.y)
                        if is_inter(inter, point, coords[i + 1]):
                            return False
        else:
            # xa = xb : usefull when the agent is placed on a resource for example.
            for i, point in enumerate(coords[:-1]):
                if point.x == coords[i + 1].x:
                    if xa == point.x:
                        inter_in_border = (min(ya, yb) <=
                                           max(point.y, coords[i + 1].y)) and \
                                          (max(ya, yb) >=
                                           min(point.y, coords[i + 1].y))
                        if inter_in_border:
                            return False
                else:
                    inter = Point(x=xa, y=point.y)
                    if is_inter(inter, point, coords[i + 1]):
                        return False
        return True

    def distance_to_resource(self, x: float, y: float, res: int, norm: str = "L2") -> float:
        if norm not in ["L1", "L2"]:
            raise ValueError("norm should be 'L1' or 'L2'.")
        diff = np.array([x - self.hp.cst_env.resources[res].x, y - self.hp.cst_env.resources[res].y])
        if norm == "L1":
            dist = np.linalg.norm(diff, ord=1)
        elif norm == "L2":
            dist = np.linalg.norm(diff)
        return dist

    def is_near_resource(self, x: float, y: float, res: int) -> bool:
        dist = self.distance_to_resource(x, y, res)
        radius = self.hp.cst_env.resources[res].r**2
        return dist < radius

    def is_resource_visible(self, x: float, y: float, res: int) -> bool:
        xb = self.hp.cst_env.resources[res].x
        yb = self.hp.cst_env.resources[res].y
        return self.is_segment_inside(x, xb, y, yb)

    def plot_resources(self, ax, scale: int, resources: List[int]=[0, 1, 2, 3]):
        """Add circles representing the resources on the plot."""
        for c_i, resource in enumerate(self.hp.cst_env.resources):
            if c_i in resources:
                x = resource.x * scale
                y = resource.y * scale
                r = resource.r * scale
                color = resource.color
                patch_circle = Circle((x, y), r, color=color)

                ax.add_patch(patch_circle)
                resource_name = f"resource_{c_i}"
                ax.text(x, y, resource_name)

    def plot(self, ax=None, save_fig: bool = False):
        """Plot the environment, but not the Agent.
        """
        if ax is None:
            ax = plt.subplot(111)

        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.hp.cst_env.coord_env + [self.hp.cst_env.coord_env[0]]

        for i, point in enumerate(coords[:-1]):
            ax.plot([point.x, coords[i + 1].x],
                    [point.y, coords[i + 1].y],
                    '-', color='black', lw=2)

        self.plot_resources(ax, scale=1)

        ax.axis('off')

        if save_fig:
            ax.savefig('environment.eps', format='eps')

