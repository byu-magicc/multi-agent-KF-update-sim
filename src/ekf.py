import numpy as np


def _g(u_t, mu_t_1):
    """
    Odometry propagation model. Given the previous state and the current control input,
    return the predicted state.

    x_t = x_t-1 + trans * cos(psi_t-1 + rot_1)
    y_t = y_t-1 + trans * sin(psi_t-1 + rot_1)
    psi_t = psi_t-1 + rot_1 + rot_2

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t.
        [[rot_1, trans, rot_2]].T
    mu_t_1: np.array, shape (3, 1)
        Previous state estimate.
        [[x, y, psi]].T

    Returns:
    mu_t: np.array, shape (3, 1)
        Predicted state estimate.
    """

    x_t = mu_t_1.item(0) + u_t.item(1) * np.cos(mu_t_1.item(2) + u_t.item(0))
    y_t = mu_t_1.item(1) + u_t.item(1) * np.sin(mu_t_1.item(2) + u_t.item(0))
    psi_t = mu_t_1.item(2) + u_t.item(0) + u_t.item(2)

    return np.array([x_t, y_t, psi_t]).reshape(-1, 1)


def _G_mu(u_t, mu_t_1):
    """
    Jacobian of the odometry propagation model with respect to the state.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. [[rot_1, trans, rot_2]].T
    mu_t_1: np.array, shape (3, 1)
        State estimate at time t-1. [[x, y, psi]].T

    Returns:
    G_mu_t: np.array, shape (3, 3)
    """

    G_mu_t = np.array([[1, 0, -u_t.item(1) * np.sin(mu_t_1.item(2) + u_t.item(0))],
                       [0, 1, u_t.item(1) * np.cos(mu_t_1.item(2) + u_t.item(0))],
                       [0, 0, 1]])

    return G_mu_t


def _G_u(u_t, mu_t_1):
    """
    Jacobian of the odometry propagation model with respect to input.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. [[rot_1, trans, rot_2]].T
    mu_t_1: np.array, shape (3, 1)
        Current state estimate. [[x, y, psi]].T

    Returns:
    G_u_t: np.array, shape (3, 3)
    """

    G_u_t = np.array([[-u_t.item(1) * np.sin(mu_t_1.item(2) + u_t.item(0)),
                       np.cos(mu_t_1.item(2) + u_t.item(0)), 0],
                      [u_t.item(1) * np.cos(mu_t_1.item(2) + u_t.item(0)),
                       np.sin(mu_t_1.item(2) + u_t.item(0)), 0],
                      [1, 0, 1]])

    return G_u_t


def _h_global(mu_t):
    """
    Global measurement model. Given the current state, return the predicted global pose.

    Parameters:
    mu_t: np.array, shape (3, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi]].T

    Returns:
    np.array, shape (3, 1)
        Predicted global pose.
    """

    return mu_t


def _H_global(mu_t):
    """
    Jacobian of the global measurement model with respect to the state.

    Parameters:
    mu_t: np.array, shape (3, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi]].T

    Returns:
    H_global_t: np.array, shape (3, 3)
        Jacobian of the global measurement model.
    """

    return np.eye(3, 3)


class EKF:
    """
    Basic EKF class for odometry propagation and global measurements.
    """
    def __init__(self, mu_0, Sigma_0, alphas):
        """
        Parameters:
        mu_0: np.array, shape (3, 1)
            Initial state estimate. Contains the position and orientation.
            [[x, y, psi]].T
        Sigma_0: np.array, shape (3, 3)
            Initial covariance matrix.
        alphas: odometry noise coefficients
        """
        assert mu_0.shape == (3, 1)
        assert Sigma_0.shape == (3, 3)

        self.mu = mu_0
        self.Sigma = Sigma_0
        self.alphas = alphas

        self.g = _g
        self.h_global = _h_global
        self.G_mu = _G_mu
        self.G_u = _G_u
        self.H_global = _H_global


    def propagate(self, u_t):
        """
        Propagate the state estimate and covariance matrix using the IMU measurement model.

        Parameters:
        u_t: np.array, shape (3, 1)
            Control input (odometry) at time t.
            [[rot_1, trans, rot_2]].T
        """
        assert u_t.shape == (3, 1)

        # Calculate odometry covariance
        p1 = self.alphas.item(0)*u_t.item(0)**2 + self.alphas.item(1)*u_t.item(1)**2
        p2 = self.alphas.item(2)*u_t.item(1)**2 + self.alphas.item(3)*u_t.item(0)**2 \
            + self.alphas.item(3)*u_t.item(2)**2
        p3 = self.alphas.item(0)*u_t.item(2)**2 + self.alphas.item(1)*u_t.item(1)**2
        R = np.array([[p1, 0, 0], [0, p2, 0], [0, 0, p3]])

        # Calculate Jacobians
        G_mu_t = self.G_mu(u_t, self.mu)
        G_u_t = self.G_u(u_t, self.mu)

        self.mu = self.g(u_t, self.mu)
        self.Sigma = G_mu_t @ self.Sigma @ G_mu_t.T + G_u_t @ R @ G_u_t.T

    def update_global(self, z_t, Q):
        """
        Apply a global measurement to the state estimate and covariance matrix.

        Parameters:
        z_t: np.array, shape (3, 1)
            Global measurement. [[x, y, psi]].T
        Q: np.array, shape (3, 3)
            Covariance matrix for the global measurement uncertainty.
        """
        assert z_t.shape == (3, 1)
        assert Q.shape == (3, 3)

        H_t = self.H_global(self.mu)

        K_t = self.Sigma @ H_t.T @ np.linalg.inv(H_t @ self.Sigma @ H_t.T + Q)
        self.mu = self.mu + K_t @ (z_t - self.h_global(self.mu))
        self.Sigma = (np.eye(3) - K_t @ H_t) @ self.Sigma

    def reset_state(self):
        """
        Resets the position and heading states and associated covariances to 0 and returns latest
        values before resetting. Useful for keyframe resets.

        Returns:
        current_mu: np.array, shape (3, 1)
            Current state estimate before reset.
        current_Sigma: np.array, shape (3, 3)
            Current covariance matrix before reset.
        """

        current_mu = self.mu.copy()
        current_Sigma = self.Sigma.copy()
        self.mu = np.zeros_like(self.mu)
        self.Sigma = np.zeros_like(self.Sigma)

        return current_mu, current_Sigma


if __name__ == "__main__":
    from measurements import get_odom_data
    from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
    from trajectories import sine_trajectory, line_trajectory


    np.set_printoptions(linewidth=np.inf)
    np.random.seed(0)

    num_steps = 100
    alphas = np.array([1, 1, 1, 1]) * 1e-2**2

    trajectory = sine_trajectory(num_steps, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 5, 2)

    odom_data = get_odom_data(trajectory, alphas)

    mu_0 = trajectory[:, 0].reshape(-1, 1).copy()
    Sigma_0 = np.eye(3) * 1e-15
    Sigma_global = np.diag([0.1, 0.1, 0.05])**2

    mu_hist = {"Vehicle 1": [mu_0]}
    truth_hist = {"Vehicle 1": [trajectory]}
    Sigma_hist = {"Vehicle 1": [Sigma_0.copy()]}

    ekf = EKF(mu_0, Sigma_0, alphas)

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
    plot_trajectory_error(mu_hist, truth_hist, Sigma_hist)

