from enum import Enum
import numpy as np

from ekf import EKF
from measurements import get_odom_data
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
    def __init__(self, initial_position, final_position, initial_sigmas, trajectory_type):
        """
        Parameters:
        initial_position: np.array, shape (2, 1)
            Initial position of the vehicle [x, y] in world frame.
        final_position: np.array, shape (2, 1)
            Final position of the vehicle [x, y] in world frame.
        initial_sigmas: np.array, shape (3, 1)
            Initial uncertainty of the vehicle's position, in standard deviations.
        trajectory_type: TrajectoryType
            Type of trajectory to follow.
        """
        assert initial_position.shape == (2, 1)
        assert final_position.shape == (2, 1)
        assert initial_sigmas.shape == (3, 1)
        assert trajectory_type in TrajectoryType

        # Generate trajectory
        total_distance = np.linalg.norm(final_position - initial_position)
        self._total_steps = int(total_distance)
        if trajectory_type == TrajectoryType.LINE:
            trajectory = line_trajectory(self._total_steps,
                                         initial_position,
                                         final_position)
        elif trajectory_type == TrajectoryType.ARC:
            trajectory = arc_trajectory(self._total_steps,
                                        initial_position,
                                        final_position,
                                        np.deg2rad(15))
        else:
            trajectory = sine_trajectory(self._total_steps,
                                         initial_position,
                                         final_position,
                                         total_distance / 30,
                                         3)

        # Generate odometry data
        odom_sigmas = np.array([0.1, 0.1, 0.0025]).reshape(-1, 1)
        self._odom_data = get_odom_data(trajectory, odom_sigmas)

        # Initialize EKF
        mu_0 = trajectory[:, 0].reshape(-1, 1).copy() + np.random.normal(0, initial_sigmas)
        Sigma_0 = np.diag(initial_sigmas.squeeze()**2)
        self._ekf = EKF(mu_0, Sigma_0, odom_sigmas)

        # Initialize keyframe EKF
        keyframe_mu_0 = np.zeros((3,1))
        keyframe_Sigma_0 = np.zeros((3,3))
        self._keyframe_ekf = EKF(keyframe_mu_0, keyframe_Sigma_0, odom_sigmas)

        # Initialize history
        self._mu_hist = [mu_0]
        self._truth_hist = trajectory
        self._Sigma_hist = [Sigma_0]

        # Other variables
        self._current_step = 0

    def step(self):
        """
        Increment the vehicle's simulation by one timestep.

        Returns:
        mu: np.array, shape (3, 1)
            Current state estimate. [x, y, theta]
        Sigma: np.array, shape (3, 3)
            Current state covariance.
        """
        if self._current_step >= self._odom_data.shape[1]:
            return None

        self._ekf.propagate(self._odom_data[:, self._current_step].reshape(-1, 1))
        #self._keyframe_ekf.propagate(self._odom_data[:, self._current_step].reshape(-1, 1))
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
        mu: np.array, shape (3, 1)
            Updated state estimate. [x, y, theta]
        Sigma: np.array, shape (3, 3)
            Updated state covariance.
        """
        self._ekf.update_global(z_t, sigma_z)

        return self._ekf.mu, self._ekf.Sigma

    def get_history(self):
        """
        Get the estimate, truth and covariance history of the vehicle.

        Returns:
        mu_hist: np.array, shape (3, current_step)
            History of state estimates.
        truth_hist: np.array, shape (3, current_step)
            History of true states.
        Sigma_hist: np.array, shape (current_step, 3, 3)
            History of state covariances.
        """
        return np.hstack(self._mu_hist), \
            self._truth_hist[:, :(self._current_step + 1)], \
            np.array(self._Sigma_hist)

    def get_current_step(self):
        """
        Get the current timestep of the simulation.

        Returns:
        int
            Current timestep
        """
        return self._current_step

    def is_active(self):
        """
        Reports if vehicle has finished it's trajectory.

        Returns:
        bool
            True if vehicle is not at end of trajectory, false otherwise.
        """
        return self._current_step < self._odom_data.shape[1]

    def keyframe_reset(self):
        """
        Resets keyframe EKF and returns the keyframe state and covariance prior to the reset.
        """
        return self._keyframe_ekf.reset_state()


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    initial_position = np.array([[0, 0]]).T
    final_position = np.array([[100, 100]]).T
    trajectory_type = TrajectoryType.ARC
    initial_sigmas = np.array([0.5, 0.5, 1e-2]).reshape(-1, 1)
    global_sigmas = np.array([0.5, 0.5, 1e15]).reshape(-1, 1)

    vehicle = Vehicle(initial_position, final_position, initial_sigmas, trajectory_type)

    while vehicle.step() is not None:
        if vehicle.get_current_step() % 75 == 0 and vehicle.get_current_step() != 0:
            global_meas = vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1).copy()
            global_meas += np.random.normal(0, global_sigmas)
            vehicle.update(global_meas, np.diag(global_sigmas.flatten())**2)

    mu_hist, truth_hist, Sigma_hist = vehicle.get_history()

    plot_overview(trajectories=[Trajectory(truth_hist[:2], name="Truth", color='r'),
                                Trajectory(mu_hist[:2], name="Estimate", color='b')],
                  covariances=[Covariance(vehicle._ekf.Sigma[:2, :2], vehicle._ekf.mu[:2],
                                          color="b")]
    )
    plot_trajectory_error({"Vehicle 1": [mu_hist]}, {"Vehicle 1": [truth_hist]},
                          {"Vehicle 1": [Sigma_hist]})

