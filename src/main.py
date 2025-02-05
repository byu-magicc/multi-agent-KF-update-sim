#!/usr/bin/env python3

import numpy as np

from plotters import plot_xy_trajectories
from trajectories import sine_trajectory


poses = [
    [sine_trajectory(100, np.array([0, 0]), np.array([100, 100]), 10, 5)[:2, :], "", "b"],
    [sine_trajectory(100, np.array([0, 0]), np.array([100, 0]), 10, 5)[:2, :], "Traj", ""]
]

lines = [
    [np.array([[0, 0], [100, 100]]).T, ""],
    [np.array([[0, 0], [100, 0]]).T, "r"]
]

markers = [
    [np.array([[50], [50]]), "g"],
    [np.array([[50], [50]]), ""]
]

covariances = [
    [np.array([[10, 0], [10, 1]]), np.array([[20], [50]]), "b"],
    [np.array([[12, -7], [-7, 6]]), np.array([[50], [20]]), "k"]
]

plot_xy_trajectories(poses=poses, lines=lines, markers=markers, covariances=covariances)

