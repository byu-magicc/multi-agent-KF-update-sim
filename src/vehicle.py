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
    def __init__(self, initial_position, final_position, initial_sigmas, trajectory_type, velocity):
        """
        Parameters:
        initial_position: np.array, shape (2, 1)
            Initial position of the vehicle [x, y] in world frame.
        final_position: np.array, shape (2, 1)
            Final position of the vehicle [x, y] in world frame.
        initial_sigmas: np.array, shape (5, 1)
            Initial uncertainty of the vehicle's estimate, in standard deviations.
        trajectory_type: TrajectoryType
            Type of trajectory to follow.
        velocity: float
            Velocity to travel trajectory
        """
        assert initial_position.shape == (2, 1)
        assert final_position.shape == (2, 1)
        assert initial_sigmas.shape == (5, 1)
        assert trajectory_type in TrajectoryType

        # Generate trajectory
        IMU_FREQUENCY = 100  # 1/s
        self._DT = 1 / IMU_FREQUENCY
        total_distance = np.linalg.norm(final_position - initial_position)
        self._total_steps = int(total_distance / velocity * IMU_FREQUENCY)
        if trajectory_type == TrajectoryType.LINE:
            trajectory = line_trajectory(self._total_steps,
                                         initial_position,
                                         final_position)
        elif trajectory_type == TrajectoryType.ARC:
            trajectory = arc_trajectory(self._total_steps,
                                        initial_position,
                                        final_position,
                                        np.deg2rad(45))
        else:
            trajectory = sine_trajectory(self._total_steps,
                                         initial_position,
                                         final_position,
                                         total_distance / 30,
                                         3)

        # Generate imu data
        self._IMU_SIGMAS = np.array([0.9, 0.9, np.deg2rad(0.025)]).reshape(-1, 1)
        v_0 = ((trajectory[:2, 1] - trajectory[:2, 0]) / self._DT).reshape(-1, 1)
        self._imu_data, v_truth = get_imu_data(trajectory, self._IMU_SIGMAS, v_0, self._DT)

        # Initialize EKF
        mu_0 = np.vstack((trajectory[:, 0].reshape(-1, 1).copy(), v_0)) \
            + np.random.normal(0, initial_sigmas)
        Sigma_0 = np.diag(initial_sigmas.flatten()**2)
        self._ekf = EKF(mu_0, Sigma_0, self._IMU_SIGMAS, self._DT)

        # Initialize history
        self._mu_hist = [mu_0]
        self._truth_hist = np.vstack((trajectory, v_truth))
        self._Sigma_hist = [Sigma_0]
        self._time_hist = np.linspace(0, self._DT*(trajectory.shape[1] - 1), trajectory.shape[1])

        # Other variables
        self._current_step = 0

    def step(self):
        """
        Increment the vehicle's simulation by one timestep.

        Returns:
        mu: np.array, shape (5, 1)
            Current state estimate. [x, y, theta, v_x, v_y]
        Sigma: np.array, shape (5, 3)
            Current state covariance.
        """
        if self._current_step >= self._imu_data.shape[1]:
            return None

        self._ekf.propagate(self._imu_data[:, self._current_step].reshape(-1, 1))
        self._current_step += 1

        self._mu_hist.append(self._ekf.mu.copy())
        self._Sigma_hist.append(self._ekf.Sigma.copy())

        return self._ekf.mu, self._ekf.Sigma

    def global_update(self, z_t, sigma_z):
        self._ekf.global_update(z_t, sigma_z)
        return self._ekf.mu, self._ekf.Sigma

    def range_update(self, z_t, Q, x_b, Sigma_b):
        self._ekf.range_update(z_t, Q, x_b, Sigma_b)
        return self._ekf.mu, self._ekf.Sigma

    def get_current_estimate(self):
        """
        Returns the current estimate mean and covariance
        """
        return self._ekf.mu, self._ekf.Sigma

    def get_history(self):
        """
        Get the estimate, truth and covariance history of the vehicle.

        Returns:
        time_hist: np.array, shape (current_step)
            Times associated with history arrays.
        truth_hist: np.array, shape (5, current_step)
            History of true states.
        mu_hist: np.array, shape (5, current_step)
            History of state estimates.
        Sigma_hist: np.array, shape (current_step, 5, 5)
            History of state covariances.
        """
        return self._time_hist[:(self._current_step + 1)], \
            self._truth_hist[:, :(self._current_step + 1)], \
            np.hstack(self._mu_hist), \
            np.array(self._Sigma_hist)

    def get_current_time(self):
        """
        Get the current timestep of the simulation.

        Returns:
        int
            Current timestep
        """
        return self._current_step * self._DT

    def is_active(self):
        """
        Reports if vehicle has finished it's trajectory.

        Returns:
        bool
            True if vehicle is not at end of trajectory, false otherwise.
        """
        return self._current_step < self._imu_data.shape[1]


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    initial_position = np.array([[0, 0]]).T
    final_position = np.array([[1000, 1000]]).T
    trajectory_type = TrajectoryType.SINE
    initial_sigmas = np.array([0.5, 0.5, 1e-2, 0.1, 0.1]).reshape(-1, 1)
    global_sigmas = np.array([0.5, 0.5, 1e9, 1e9, 1e9]).reshape(-1, 1)

    vehicle = Vehicle(initial_position, final_position, initial_sigmas, trajectory_type, 15)

    while vehicle.step() is not None:
        if vehicle.get_current_time() % 50 == 0 and vehicle.get_current_time() != 0:
            global_meas = vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1).copy()
            global_meas[:2] += np.random.normal(0, global_sigmas[:2])
            vehicle.global_update(global_meas, np.diag(global_sigmas.flatten())**2)

    time_hist, truth_hist, mu_hist, Sigma_hist = vehicle.get_history()

    plot_overview(trajectories=[Trajectory(truth_hist[:2], name="Truth", color='r'),
                                Trajectory(mu_hist[:2], name="Estimate", color='b')],
                  covariances=[Covariance(vehicle._ekf.Sigma[:2, :2], vehicle._ekf.mu[:2],
                                          color="b")]
    )
    plot_trajectory_error(
        time_hist,
        {"Vehicle 1": [truth_hist]},
        {"Vehicle 1": [mu_hist]},
        {"Vehicle 1": [Sigma_hist]},
    )

