from enum import Enum
import numpy as np

from ekf import EKF
from measurements import get_imu_data
from trajectories import line_trajectory, arc_trajectory, sine_trajectory


class TrajectoryType(Enum):
    LINE = 1
    ARC = 2
    SINE = 3


class Vehicle:
    def __init__(self, initial_pose, velocity, trajectory_type, total_time):
        assert initial_pose.shape == (3, 1)
        assert trajectory_type in TrajectoryType

        # Constants
        IMU_FREQUENCY = 200  # Hz
        IMU_NOISE_STD = np.array([[0.9, 0.9, np.deg2rad(0.13)]]).T  # a_x m/s^2, a_y m/s^2, omega rad/s
        self._DT = 1.0 / IMU_FREQUENCY

        # Generate trajectory
        num_steps = int(total_time * IMU_FREQUENCY)
        final_position = np.array([[
            initial_pose.item(0) + velocity * total_time * np.cos(initial_pose.item(2)),
            initial_pose.item(1) + velocity * total_time * np.sin(initial_pose.item(2))]]).T
        total_distance = np.linalg.norm(final_position - initial_pose[:2])
        if trajectory_type == TrajectoryType.LINE:
            trajectory = line_trajectory(num_steps,
                                              initial_pose[:2],
                                              final_position)
        elif trajectory_type == TrajectoryType.ARC:
            trajectory = arc_trajectory(num_steps,
                                             initial_pose[:2],
                                             final_position,
                                             np.deg2rad(15))
        else:
            trajectory = sine_trajectory(num_steps,
                                              initial_pose[:2],
                                              final_position,
                                              total_distance / 30,
                                              3)

        # Generate IMU data
        self._imu_data, v_0 = get_imu_data(trajectory, IMU_NOISE_STD, self._DT)

        # Initialize EKF
        mu_0 = np.vstack([trajectory[:, 0].reshape(-1, 1).copy(), v_0])
        Sigma_0 = np.eye(5) * 1e-9
        self._Sigma_imu = np.diag(IMU_NOISE_STD.squeeze()**2)
        self._ekf = EKF(mu_0, Sigma_0)

        # Initialize history
        self._mu_hist = [mu_0]
        # TODO: Add velocity to truth_hist (also update plotter)
        self._truth_hist = np.pad(trajectory, ((0, 2), (0, 0)), mode='constant', constant_values=0)
        self._Sigma_hist = [Sigma_0]

        # Other variables
        self._current_step = 0

    def step(self):
        if self._current_step >= self._imu_data.shape[1]:
            return None

        self._ekf.propagate(self._imu_data[:, self._current_step].reshape(-1, 1), self._Sigma_imu, self._DT)
        self._current_step += 1

        self._mu_hist.append(self._ekf.mu.copy())
        self._Sigma_hist.append(self._ekf.Sigma.copy())

        return self._ekf.mu, self._ekf.Sigma

    def update(self, z_t, sigma_z):
        assert z_t.shape == (3, 1)
        assert sigma_z.shape == (3, 3)

        self._ekf.update_global(z_t, sigma_z)

        return self._ekf.mu, self._ekf.Sigma

    def get_history(self):
        return self._mu_hist, self._truth_hist, self._Sigma_hist

    def get_current_time(self):
        return self._current_step * self._DT


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview

    np.set_printoptions(linewidth=np.inf)
    np.random.seed(1)

    total_time = 60
    initial_pose = np.array([[0, 0, np.deg2rad(45)]]).T
    velocity = 15
    trajectory_type = TrajectoryType.ARC

    vehicle = Vehicle(initial_pose, velocity, trajectory_type, total_time)

    while vehicle.step() is not None:
        if vehicle.get_current_time() % 120 == 0:
            global_meas = vehicle._truth_hist[:3, vehicle._current_step].reshape(-1, 1)
            vehicle.update(global_meas, np.diag([0.5, 0.5, np.inf])**2)

    mu_hist, truth_hist, Sigma_hist = vehicle.get_history()

    mu_hist = np.hstack(mu_hist)[:3, :]
    truth_hist = truth_hist[:3, :]
    Sigma_hist = np.hstack([np.sqrt(Sigma[:3, :3].diagonal().reshape(-1, 1)) for Sigma in Sigma_hist])

    plot_overview(poses=[[truth_hist[:2], "Truth", "r"], [mu_hist[:2], "Estimate", "b"]],
                         covariances=[[vehicle._ekf.Sigma[:2, :2], vehicle._ekf.mu[:2], "b"]])
    plot_trajectory_error({"Vehicle 1": mu_hist}, {"Vehicle 1": truth_hist}, {"Vehicle 1": Sigma_hist})

