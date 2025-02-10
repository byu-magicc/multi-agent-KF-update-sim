#!/usr/bin/env python3

import numpy as np

from measurements import get_imu_data
from plotters import plot_xy_trajectories
from trajectories import line_trajectory, arc_trajectory, sine_trajectory

import matplotlib.pyplot as plt


time = 60
dt = 1.0 / 400
num_steps = int(time / dt)
noise_std = np.array([[0., 0., 0.]]).T

x = np.array([[0, 0]], dtype=float).T

trajectory = line_trajectory(num_steps, x, np.array([[10, 10]]).T)
trajectory = arc_trajectory(num_steps, x, np.array([[10, 0]]).T, 10)
trajectory = sine_trajectory(num_steps, x, np.array([[100, 100]]).T, 5, 2)
imu_data, v = get_imu_data(trajectory, noise_std, dt)

x_traj = [trajectory[:, 0].reshape(-1, 1).copy()]
x = trajectory[:, 0].reshape(-1, 1).copy()

for i in range(num_steps - 1):
    rot = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]]).squeeze()

    x[:2] += v * dt + 0.5 * rot @ imu_data[:2, i].reshape(-1, 1) * dt ** 2
    x[2] += imu_data[2, i] * dt
    v += rot @ imu_data[:2, i].reshape(-1, 1) * dt

    x_traj.append(x.copy())

    print(f"Step {i}: x = {x.T}, v = {v.T}, imu = {imu_data[:, i]}")

x_traj = np.hstack(x_traj)

plot_xy_trajectories([[trajectory[:2], "Truth", "r"], [x_traj[:2], "Estimate", "b"]])

error = x_traj[:2] - trajectory[:2]
error_psi = x_traj[2] - trajectory[2]
plt.plot(error[0, :], label="x error")
plt.plot(error[1, :], label="y error")
plt.plot(error_psi, label="psi error")
plt.legend()
plt.show()
