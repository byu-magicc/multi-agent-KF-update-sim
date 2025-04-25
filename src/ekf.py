import numpy as np


def _g(u_t, mu_t_1):
    """
    Odometry propagation model. Given the previous state and the current control input,
    return the predicted state.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t.
        [[delta_x, delta_y, delta_theta]].T
    mu_t_1: np.array, shape (3, 1)
        Previous state estimate.
        [[x, y, theta]].T

    Returns:
    mu_t: np.array, shape (3, 1)
        Predicted state estimate.
    """

    theta_t_1 = mu_t_1[2]
    R = np.array([[np.cos(theta_t_1), -np.sin(theta_t_1)],
                  [np.sin(theta_t_1), np.cos(theta_t_1)]]).squeeze()
    xy_t = R @ u_t[:2] + mu_t_1[:2]
    theta_t = mu_t_1[2] + u_t[2]

    return np.vstack([xy_t, theta_t])


def _G_mu(u_t, mu_t_1):
    """
    Jacobian of the odometry propagation model with respect to the state.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. [[delta_x, delta_y, delta_theta]].T
    mu_t_1: np.array, shape (3, 1)
        State estimate at time t-1. [[x, y, theta]].T

    Returns:
    G_mu_t: np.array, shape (3, 3)
    """

    theta_t_1 = mu_t_1.item(2)
    G_mu_t = np.array([[1, 0, -u_t.item(0) * np.sin(theta_t_1) - u_t.item(1) * np.cos(theta_t_1)],
                       [0, 1, u_t.item(0) * np.cos(theta_t_1) - u_t.item(1) * np.sin(theta_t_1)],
                       [0, 0, 1]])

    return G_mu_t


def _G_u(u_t, mu_t_1):
    """
    Jacobian of the odometry propagation model with respect to input.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. [[delta_x, delta_y, delta_theta]].T
    mu_t_1: np.array, shape (3, 1)
        State estimate at time t-1. [[x, y, theta]].T

    Returns:
    G_u_t: np.array, shape (3, 3)
    """

    theta_t_1 = mu_t_1.item(2)
    G_u_t = np.array([[np.cos(theta_t_1), -np.sin(theta_t_1), 0],
                      [np.sin(theta_t_1), np.cos(theta_t_1), 0],
                      [0, 0, 1]])

    return G_u_t


def _h_global(mu_t):
    """
    Global measurement model. Given the current state, return the predicted global pose.

    Parameters:
    mu_t: np.array, shape (3, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, theta]].T

    Returns:
    np.array, shape (3, 1)
        Predicted global pose.
    """

    return mu_t



def _h_shared_GPS(mu_t, z_gps, T_b_a):
    """
    Measurement model for shared GPS update. Doesn't return predicted measurement, but measurement
    residual.

    Parameters:
    mu_t: np.array, shape (3, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, theta]].T
    z_gps: np.array, shape (2, 1)
        GPS measurement. [[x, y]].T
    T_b_a: np.array, shape (2, 1)
        Translation from vehicle b to vehicle a, in frame of vehicle b. [[delta_x, delta_y]].T

    Returns:
    np.array, shape (2, 1)
        Measurement residual.
    """
    theta = mu_t.item(2)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]).squeeze()

    return z_gps - (mu_t[:2] + R @ T_b_a)


def _H_global(mu_t):
    """
    Jacobian of the global measurement model with respect to the state.

    Parameters:
    mu_t: np.array, shape (3, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, theta]].T

    Returns:
    H_global_t: np.array, shape (3, 3)
        Jacobian of the global measurement model.
    """

    return np.eye(3, 3)


def _H_shared_GPS(theta_t, T_b_a):
    """
    Jacobian of the shared GPS measurement with respect to the state.

    Parameters
    theta: float
        Current theta estimate.
    T_b_a: np.array, shape (2, 1)
        Translation from vehicle b to vehicle a, in frame of vehicle b. [[delta_x, delta_y]].T

    Returns:
    H_shared_GPS: np.array, shape (2, 2)
        Jacobian of the shared GPS measurement.
    """
    assert T_b_a.shape == (2, 1)

    H_shared_GPS = np.array([[1, 0, -T_b_a.item(0) * np.sin(theta_t) - T_b_a.item(1) * np.cos(theta_t)],
                             [0, 1, T_b_a.item(0) * np.cos(theta_t) - T_b_a.item(1) * np.sin(theta_t)]])

    return H_shared_GPS


def _F_shared_GPS(theta_t):
    """
    Jacobian of the shared GPS measurement with respect to the measurement.

    Parameters
    theta: float
        Current theta estimate.

    Returns:
    F_shared_GPS: np.array, shape (2, 2)
        Jacobian of the shared GPS measurement.
    """
    F_shared_GPS = np.array([[-1, 0, np.cos(theta_t), -np.sin(theta_t)],
                             [0, -1, np.sin(theta_t), np.cos(theta_t)]])

    return F_shared_GPS


class EKF:
    """
    Basic EKF class for odometry propagation and global measurements.
    """
    def __init__(self, mu_0, Sigma_0, odom_sigmas):
        """
        Parameters:
        mu_0: np.array, shape (3, 1)
            Initial state estimate. Contains the position and orientation. [[x, y, theta]].T
        Sigma_0: np.array, shape (3, 3)
            Initial covariance matrix.
        odom_sigmas: np.array, shape (3, 1)
            Standard deviations of the odometry noise. [[sigma_x, sigma_y, sigma_theta]].T
        """
        assert mu_0.shape == (3, 1)
        assert Sigma_0.shape == (3, 3)
        assert odom_sigmas.shape == (3, 1)

        self.mu = mu_0
        self.Sigma = Sigma_0
        self.R = np.diag(odom_sigmas.flatten()**2)

        self.g = _g
        self.h_global = _h_global
        self.h_shared_GPS = _h_shared_GPS
        self.G_mu = _G_mu
        self.G_u = _G_u
        self.H_global = _H_global
        self.H_shared_GPS = _H_shared_GPS
        self.F_shared_GPS = _F_shared_GPS

    def propagate(self, u_t):
        """
        Propagate the state estimate and covariance matrix.

        Parameters:
        u_t: np.array, shape (3, 1)
            Control input (odometry) at time t.
            [[delta_x, delta_y, delta_theta]].T
        """
        assert u_t.shape == (3, 1)

        # Calculate Jacobians
        G_mu_t = self.G_mu(u_t, self.mu)
        G_u_t = self.G_u(u_t, self.mu)

        self.mu = self.g(u_t, self.mu)
        self.Sigma = G_mu_t @ self.Sigma @ G_mu_t.T + G_u_t @ self.R @ G_u_t.T

    def update_global(self, z_t, Q):
        """
        Apply a global measurement to the state estimate and covariance matrix.

        Parameters:
        z_t: np.array, shape (3, 1)
            Global measurement. [[x, y, theta]].T
        Q: np.array, shape (3, 3)
            Covariance matrix for the global measurement uncertainty.
        """
        assert z_t.shape == (3, 1)
        assert Q.shape == (3, 3)

        H_t = self.H_global(self.mu)

        K_t = self.Sigma @ H_t.T @ np.linalg.inv(H_t @ self.Sigma @ H_t.T + Q)
        self.mu = self.mu + K_t @ (z_t - self.h_global(self.mu))
        self.Sigma = (np.eye(3) - K_t @ H_t) @ self.Sigma

    def update_shared_gps(self, z_gps, T_b_a, Sigma_GPS, Sigma_T):
        """
        Apply a shared global measurement to the state estimate and covariance matrix.
        Shared measurements are when a different agent (a) recieves a GPS measurement and translation
        information from a multi-agent backend is used to apply that measurement to another vehicle
        (b).

        Parameters:
        z_gps: np.array, shape (2, 1)
            GPS measurement recieved at vehicle a. [[x, y]].T
        T_b_a: np.array, shape (2, 1)
            Translation information from vehicle b to a, in the frame of vehicle b.
            [[delta_x, delta_y]].T
        Sigma_GPS: np.array, shape (2, 2)
            Covariance matrix of GPS measurement, in global frame.
        Sigma_T: np.array, shape (2, 2)
            Covariance matrix of translation information, in the frame of vehicle b.
        """
        assert z_gps.shape == (2, 1)
        assert T_b_a.shape == (2, 1)
        assert Sigma_GPS.shape == (2, 2)
        assert Sigma_T.shape == (2, 2)

        res = self.h_shared_GPS(self.mu, z_gps, T_b_a)
        H_t = self.H_shared_GPS(self.mu.item(2), T_b_a)
        F_t = self.F_shared_GPS(self.mu.item(2))

        Sigma_z = np.zeros((4, 4))
        Sigma_z[:2, :2] = Sigma_GPS
        Sigma_z[2:, 2:] = Sigma_T

        S_t = H_t @ self.Sigma @ H_t.T + F_t @ Sigma_z @ F_t.T
        K_t = self.Sigma @ H_t.T @ np.linalg.inv(S_t)
        self.mu = self.mu + K_t @ res
        self.Sigma = (np.eye(3) - K_t @ H_t) @ self.Sigma


if __name__ == "__main__":
    from measurements import get_odom_data
    from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
    from trajectories import sine_trajectory, line_trajectory


    np.set_printoptions(linewidth=np.inf)
    np.random.seed(0)

    num_steps = 100
    odom_sigmas = np.array([0.1, 0.1, np.deg2rad(1)]).reshape(-1, 1)

    trajectory = sine_trajectory(num_steps, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 5, 2)
    hist_indices = np.arange(0, trajectory.shape[1])

    odom_data = get_odom_data(trajectory, odom_sigmas)

    mu_0 = trajectory[:, 0].reshape(-1, 1).copy()
    Sigma_0 = np.eye(3) * 1e-15
    Sigma_global = np.diag([0.1, 0.1, 0.05])**2

    mu_hist = {"Vehicle 1": [mu_0]}
    truth_hist = {"Vehicle 1": [trajectory]}
    Sigma_hist = {"Vehicle 1": [Sigma_0.copy()]}

    ekf = EKF(mu_0, Sigma_0, odom_sigmas)

    for i in range(num_steps - 1):
        ekf.propagate(odom_data[:, i].reshape(-1, 1))

        mu_hist["Vehicle 1"].append(ekf.mu.copy())
        Sigma_hist["Vehicle 1"].append(ekf.Sigma.copy())

        if i % 50 == 0 and i != 0:
            global_meas = trajectory[:3, i + 1].reshape(-1, 1)
            ekf.update_global(global_meas, Sigma_global)

    mu_hist["Vehicle 1"] = [np.hstack(mu_hist["Vehicle 1"])]
    Sigma_hist["Vehicle 1"] = [np.array(Sigma_hist["Vehicle 1"])]

    plot_overview(trajectories=[Trajectory(trajectory[:2], name="Truth", color="r"),
                                Trajectory(mu_hist["Vehicle 1"][0][:2], name="Estimate", color="b")],
                  covariances=[Covariance(ekf.Sigma[:2, :2], ekf.mu[:2], color="b")])

    plot_trajectory_error(hist_indices, truth_hist, mu_hist, Sigma_hist)

