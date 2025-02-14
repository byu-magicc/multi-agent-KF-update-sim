from enum import Enum
import numpy as np

from ekf import EKF
from measurements import get_imu_data
from trajectories import line_trajectory, arc_trajectory, sine_trajectory


class TrajectoryType(Enum):
    """
    Enum class for trajectory types.
    """
    LINE = 1
    ARC = 2
    SINE = 3


class Vehicle:
    """
    Class for single simulation vehicle. Contains it's own sensor and filters and 'dynamics',
    like you would expect an actual vehicle to have.
    """
    def __init__(self, initial_pose, velocity, trajectory_type, total_time):
        """
        Parameters:
        initial_pose: np.array, shape (3, 1)
            Initial pose of the vehicle [x, y, theta] in world frame.
        velocity: float
            Initial velocity of the vehicle.
        trajectory_type: TrajectoryType
            Type of trajectory to follow.
        total_time: float
            Total time of the simulation in seconds.
        """
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
        """
        Increment the vehicle's simulation by one timestep.

        Returns:
        mu: np.array, shape (5, 1)
            Current state estimate. [x, y, theta, v_x, v_y]
        Sigma: np.array, shape (5, 5)
            Current state covariance.
        """
        if self._current_step >= self._imu_data.shape[1]:
            return None

        self._ekf.propagate(self._imu_data[:, self._current_step].reshape(-1, 1), self._Sigma_imu, self._DT)
        self._current_step += 1

        self._mu_hist.append(self._ekf.mu.copy())
        self._Sigma_hist.append(self._ekf.Sigma.copy())

        return self._ekf.mu, self._ekf.Sigma

    def update(self, z_t, sigma_z):
        """
        Update the state estimate with a global measurement.

        Parameters:
        z_t: np.array, shape (3, 1)
            Global measurement [x, y, theta] in world frame.
        sigma_z: np.array, shape (3, 3)
            Covariance of the global measurement.

        Returns:
        mu: np.array, shape (5, 1)
            Updated state estimate. [x, y, theta, v_x, v_y]
        Sigma: np.array, shape (5, 5)
            Updated state covariance.
        """
        assert z_t.shape == (3, 1)
        assert sigma_z.shape == (3, 3)

        self._ekf.update_global(z_t, sigma_z)

        return self._ekf.mu, self._ekf.Sigma

    def get_history(self):
        """
        Get the estimate, truth and covariance history of the vehicle.

        Returns:
        mu_hist: np.array, shape (5, current_step)
            History of state estimates.
        truth_hist: np.array, shape (5, current_step)
            History of true states.
        Sigma_hist: np.array, shape (current_step, 5, 5)
            History of state covariances.
        """
        return np.hstack(self._mu_hist), \
            self._truth_hist[:, :(self._current_step + 1)], \
            np.array(self._Sigma_hist)

    def get_current_time(self):
        """
        Get the current time of the simulation.

        Returns:
        float
            Current time in seconds.
        """
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

    plot_overview(poses=[[truth_hist[:2], "Truth", "r"], [mu_hist[:2], "Estimate", "b"]],
                         covariances=[[vehicle._ekf.Sigma[:2, :2], vehicle._ekf.mu[:2], "b"]])
    plot_trajectory_error({"Vehicle 1": mu_hist}, {"Vehicle 1": truth_hist}, {"Vehicle 1": Sigma_hist})

