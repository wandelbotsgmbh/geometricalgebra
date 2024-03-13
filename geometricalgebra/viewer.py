"""Viewer for 3D vector and motion paths

Classes:
    PathViewer: Show path in cartesian space
    OrientationViewer: Show orientation space on unit sphere
"""

from itertools import count
from typing import Union

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy import linalg

from geometricalgebra import cga3d

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise ImportError("Please install geometricalgebra[view] to allow viewer functionality") from error


class Arrow3D(FancyArrowPatch):
    """A 3d Arrow"""

    def __init__(self, x, y, z, dx, dy, dz, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_3d_projection(self, renderer=None):  # pylint: disable=unused-argument
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _get_unit_sphere(root_num_points):
    u = np.linspace(0, 2 * np.pi, root_num_points)
    v = np.linspace(0, np.pi, root_num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return np.stack([x, y, z], axis=-1)


def _discretize_circle(circle: cga3d.Vector, num_points: int) -> np.ndarray:
    center, normal, radius = circle.circle_to_center_normal_radius()
    a, b = linalg.null_space(normal.to_euclid()[None]).T
    theta = np.linspace(0, 2 * np.pi, num_points)
    points = center.to_euclid() + radius.to_scalar() * (np.cos(theta[:, None]) * a + np.sin(theta[:, None]) * b)
    return points


class Viewer:
    """View elements of conformal geometric algebra in 3D"""

    def __init__(self, axes=None):
        """
        Args:
            axes: the axes on which to show paths (must be 3dprojective axes)
        """
        self._axes = axes or plt.figure().add_subplot(111, projection="3d")

    def __getattr__(self, key):
        return getattr(self._axes, key)

    def __dir__(self):
        return dir(self) + dir(self._axes)

    def trace_point(self, tensor: Union[list[cga3d.Vector], cga3d.Vector], linestyle="None", marker="o", **kwargs):
        if isinstance(tensor, list):
            tensor = cga3d.Vector.stack(tensor)
        return self._axes.plot(*tensor.ravel().to_euclid().T, linestyle=linestyle, marker=marker, **kwargs)

    def trace_point_pair(self, tensor: cga3d.Vector, linestyle="-", startmarker="o", endmarker=">", **kwargs):
        for pair in tensor.ravel():
            end_points = pair.point_pair_to_end_points()
            (object_,) = self.trace_point(
                end_points, linestyle=linestyle, marker="", label=kwargs.pop("label", None), **kwargs
            )
            kwargs["color"] = object_.get_color()
            self.trace_point(
                end_points[0], linestyle=linestyle, marker=startmarker, label=kwargs.pop("label", None), **kwargs
            )
            self.trace_point(
                end_points[1], linestyle=linestyle, marker=endmarker, label=kwargs.pop("label", None), **kwargs
            )

    def trace_circle(self, tensor: cga3d.Vector, num_points=1000, **kwargs):
        for center, normal, radius in zip(*tensor.ravel().circle_to_center_normal_radius()):
            a, b = linalg.null_space(normal.to_euclid()[None]).T
            theta = np.linspace(0, 2 * np.pi, num_points)
            points = center.to_euclid() + radius.to_scalar() * (np.cos(theta[:, None]) * a + np.sin(theta[:, None]) * b)
            self.plot(*points.T, **kwargs)

    def trace_line(self, tensor: cga3d.Vector, length=10, **kwargs):
        for i, point, direction in zip(count(), *tensor.ravel().line_to_point_direction()):
            eucid_point = point.to_euclid()
            instance_length = length[i] if isinstance(length, np.ndarray) else length
            offset = instance_length * direction.normed().to_direction(False)
            endpoints = np.stack([eucid_point + offset, eucid_point - offset])
            self.plot(*endpoints.T, **kwargs)

    def trace_sphere(self, sphere, alpha=0.5, root_num_points=20, allow_neg_radius=True, **kwargs):
        """Plot a sphere

        Args:
            sphere: the object(s) to show
            alpha: the transparency of the shown object
            root_num_points: the sphere is drawn by a mesh consisting of root_num_points**2 points
            allow_neg_radius: if True, negative squared radius are allows (sign is ignored)
            **kwargs: are forwarded to plt.plot_surface

        Returns:
            the figure
        """
        unit_sphere = _get_unit_sphere(root_num_points)
        center, squared_radius = sphere.sphere_to_center_squared_radius()
        radius = abs(squared_radius.to_scalar()) ** 0.5 if allow_neg_radius else squared_radius.to_scalar() ** 0.5
        points = unit_sphere * radius + center.to_euclid()
        return self.plot_surface(*points.T, alpha=alpha, **kwargs)

    def trace_plane(self, planes: cga3d.Vector, num_points=64, alpha=0.5, center=(0, 0, 0), radius=1, **kwargs):
        sphere = cga3d.Vector.from_sphere(center, radius)
        circles = planes.meet(sphere)
        for circle in circles.ravel():
            points = _discretize_circle(circle, num_points)
            circle_center = np.mean(points, axis=0, keepdims=True)
            points = np.stack(np.broadcast_arrays(circle_center, points), axis=0)
            self.plot_surface(*points.T, alpha=alpha, **kwargs)

    def trace_pose(self, motor: cga3d.Vector, linestyle="-", marker="o", length=1, style=None, **kwargs):
        basis = cga3d.Vector.stack(
            [cga3d.e_0, (length * cga3d.e_1).up(), (length * cga3d.e_2).up(), (length * cga3d.e_3).up()]
        )
        points = motor.apply(basis)
        if style == "rgb":
            for i, c in zip([1, 2, 3], "rgb"):
                arrow = Arrow3D(
                    *points[0].to_euclid(),
                    *(points[i].to_euclid() - points[0].to_euclid()),
                    arrowstyle="-|>",
                    ec=c,
                    fc=c,
                    mutation_scale=25,
                )
                self._axes.add_artist(arrow)
        else:
            (object_,) = self.trace_point(
                points[:2], linestyle=linestyle, marker=marker, label=kwargs.pop("label", None), **kwargs
            )
            kwargs["color"] = object_.get_color()
            if style is None:
                self.trace_point(
                    points[[0, 2]], linestyle=linestyle, marker="", label=kwargs.pop("label", None), **kwargs
                )
            self.trace_point(points[[0, 3]], linestyle=linestyle, marker="", label=kwargs.pop("label", None), **kwargs)

    def trace_tangent_line(self, tangent):
        start, direction = tangent.to_point_direction()
        end = cga3d.Vector.from_translator(direction).apply(start)
        self.trace_point_pair(start ^ end)

    def set_axes_equal(self):
        limits = np.array([self.get_xlim3d(), self.get_ylim3d(), self.get_zlim3d()])
        center = np.mean(limits, axis=1)
        range_ = abs(limits[:, 1] - limits[:, 0])
        plot_radius = 0.5 * max(range_)
        self.set_xlim3d([center[0] - plot_radius, center[0] + plot_radius])
        self.set_ylim3d([center[1] - plot_radius, center[1] + plot_radius])
        self.set_zlim3d([center[2] - plot_radius, center[2] + plot_radius])

    def set_axes_labels(self, x_label="x", y_label="y", z_label="z"):
        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.set_zlabel(z_label)
