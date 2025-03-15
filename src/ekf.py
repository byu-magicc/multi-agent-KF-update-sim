import numpy as np


def _g(u_t, mu_t_1, dt):
    """
    IMU propagation model. Given the previous state and the current control input,
    return the predicted state.

    x_t = x_t-1 + v_t-1 * dt + 0.5 * R_psi @ a_t * dt^2
    psi_t = psi_t-1 + omega_t * dt
    v_t = v_t-1 + R_psi @ a_t * dt

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
        [[a_x, a_y, omega]].T
    mu_t_1: np.array, shape (5, 1)
        Previous state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    mu_t: np.array, shape (5, 1)
        Predicted state estimate.
    """

    a_t = u_t[0:2]
    omega_t = u_t[2]

    p_t_1 = mu_t_1[:2]
    psi_t_1 = mu_t_1[2]
    v_t_1 = mu_t_1[3:]

    R_psi = np.array([[np.cos(psi_t_1), -np.sin(psi_t_1)],
                       [np.sin(psi_t_1), np.cos(psi_t_1)]]).squeeze()

    p_t = p_t_1 + v_t_1 * dt + 0.5 * R_psi @ a_t * dt**2
    psi_t = psi_t_1 + omega_t * dt
    v_t = v_t_1 + R_psi @ a_t * dt

    return np.vstack([p_t, psi_t, v_t])


def _G_mu(u_t, mu_t_1, dt):
    """
    Jacobian of the IMU propagation model with respect to the state.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
        [[a_x, a_y, omega]].T
    mu_t_1: np.array, shape (5, 1)
        State estimate at time t-1. Contains the position, orientation, and velocity.
        [[x, y, psi, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    G_mu_t: np.array, shape (5, 5)
    """

    a_t = u_t[0:2]
    psi_t_1 = mu_t_1[2]

    R_prime = np.array([[-np.sin(psi_t_1), -np.cos(psi_t_1)],
                        [np.cos(psi_t_1), -np.sin(psi_t_1)]]).squeeze()

    p_prime_psi = 0.5 * R_prime @ a_t * dt**2
    v_prime_psi = R_prime @ a_t * dt

    G_mu_t = np.array([[1, 0, p_prime_psi.item(0), dt, 0],
                       [0, 1, p_prime_psi.item(1), 0, dt],
                       [0, 0, 1, 0, 0],
                       [0, 0, v_prime_psi.item(0), 1, 0],
                       [0, 0, v_prime_psi.item(1), 0, 1]])

    return G_mu_t


def _G_u(u_t, mu_t_1, dt):
    """
    Jacobian of the IMU propagation model with respect to input.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
        [[a_x, a_y, omega]].T
    mu_t_1: np.array, shape (5, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    G_u_t: np.array, shape (5, 5)
    """

    psi_t_1 = mu_t_1[2]

    R_psi = np.array([[np.cos(psi_t_1), -np.sin(psi_t_1)],
                      [np.sin(psi_t_1), np.cos(psi_t_1)]]).squeeze()

    p_prime_a = 0.5 * R_psi * dt**2
    v_prime_a = R_psi * dt

    G_u_t = np.array([[p_prime_a[0, 0], p_prime_a[0, 1], 0],
                      [p_prime_a[1, 0], p_prime_a[1, 1], 0],
                      [0, 0, dt],
                      [v_prime_a[0, 0], v_prime_a[0, 1], 0],
                      [v_prime_a[1, 0], v_prime_a[1, 1], 0]])

    return G_u_t


def _h_global(mu_t):
    """
    Global measurement model. Given the current state, return the predicted global pose.

    Parameters:
    mu_t: np.array, shape (5, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi, v_x, v_y]].T

    Returns:
    np.array, shape (3, 1)
        Predicted global pose.
    """

    return mu_t[:3]


def _H_global(mu_t):
    """
    Jacobian of the global measurement model with respect to the state.

    Parameters:
    mu_t: np.array, shape (5, 1)
        Current state estimate. Contains the position, orientation, and velocity.
        [[x, y, psi, v_x, v_y]].T

    Returns:
    H_global_t: np.array, shape (3, 5)
        Jacobian of the global measurement model.
    """

    return np.eye(3, 5)


class EKF:
    """
    Basic EKF class for IMU propagation and global measurements.
    """
    def __init__(self, mu_0, Sigma_0):
        """
        Parameters:
        mu_0: np.array, shape (5, 1)
            Initial state estimate. Contains the position, orientation, and velocity.
            [[x, y, psi, v_x, v_y]].T
        Sigma_0: np.array, shape (5, 5)
            Initial covariance matrix.
        """
        assert mu_0.shape == (5, 1)
        assert Sigma_0.shape == (5, 5)

        self.mu = mu_0
        self.Sigma = Sigma_0

        self.g = _g
        self.h_global = _h_global
        self.G_mu = _G_mu
        self.G_u = _G_u
        self.H_global = _H_global

    def propagate(self, u_t, R, dt):
        """
        Propagate the state estimate and covariance matrix using the IMU measurement model.

        Parameters:
        u_t: np.array, shape (3, 1)
            Control input at time t. Contains the acceleration and angular velocity.
            [[a_x, a_y, omega]].T
        R: np.array, shape (3, 3)
            Covariance matrix for the IMU measurement uncertainty.
        dt: float
            Time step.
        """
        assert u_t.shape == (3, 1)
        assert R.shape == (3, 3)

        G_mu_t = self.G_mu(u_t, self.mu, dt)
        G_u_t = self.G_u(u_t, self.mu, dt)

        self.mu = self.g(u_t, self.mu, dt)
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
        self.Sigma = (np.eye(5) - K_t @ H_t) @ self.Sigma

    def reset_state(self):
        """
        Resets the position and heading states and associated covariances to 0 and returns latest
        values before resetting. Useful for keyframe resets.

        Returns:
        current_mu: np.array, shape (5, 1)
            Current state estimate before reset.
        current_Sigma: np.array, shape (5, 5)
            Current covariance matrix before reset.
        """
        rot = np.array([[np.cos(self.mu[2]), -np.sin(self.mu[2])],
                        [np.sin(self.mu[2]), np.cos(self.mu[2])]])

        current_mu = self.mu.copy()
        current_Sigma = self.Sigma.copy()
        self.mu[:3] = 0.0
        self.mu[3:] = rot.T @ self.mu[3:]
        self.Sigma = np.eye(5) * 1e-9
        self.Sigma[3:, 3:] = current_Sigma[3:, 3:].copy()

        return current_mu[:3], current_Sigma[:3, :3]


if __name__ == "__main__":
    from measurements import get_imu_data
    from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
    from trajectories import sine_trajectory


    np.set_printoptions(linewidth=np.inf)
    np.random.seed(0)

    total_time = 60
    dt = 1.0 / 200
    num_steps = int(total_time / dt)
    imu_noise_std = np.array([[0.5, 0.5, 0.25]]).T

    trajectory = sine_trajectory(num_steps, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 5, 2)

    imu_data, v_0 = get_imu_data(trajectory, imu_noise_std, dt)

    mu_0 = np.vstack([trajectory[:, 0].reshape(-1, 1).copy(), v_0])
    Sigma_0 = np.eye(5) * 1e-9
    Sigma_global = np.diag([0.1, 0.1, 0.05])**2
    Sigma_imu = np.diag(imu_noise_std.squeeze()**2)

    mu_hist = {"Vehicle 1": [mu_0]}
    truth_hist = {"Vehicle 1": np.pad(trajectory, ((0, 2), (0, 0)), mode='constant', constant_values=0)}
    Sigma_hist = {"Vehicle 1": [Sigma_0.copy()]}

    ekf = EKF(mu_0, Sigma_0)

    for i in range(num_steps - 1):
        ekf.propagate(imu_data[:, i].reshape(-1, 1), Sigma_imu, dt)

        mu_hist["Vehicle 1"].append(ekf.mu.copy())
        Sigma_hist["Vehicle 1"].append(ekf.Sigma.copy())

        if ((i + 1) * dt) % 20 == 0:
            global_meas = trajectory[:3, i].reshape(-1, 1)
            ekf.update_global(global_meas, Sigma_global)

    mu_hist["Vehicle 1"] = np.hstack(mu_hist["Vehicle 1"])
    Sigma_hist["Vehicle 1"] = np.array(Sigma_hist["Vehicle 1"])

    plot_overview(trajectories=[Trajectory(trajectory[:2], name="Truth", color="r"),
                                Trajectory(mu_hist["Vehicle 1"][:2], name="Estimate", color="b")],
                  covariances=[Covariance(ekf.Sigma[:2, :2], ekf.mu[:2], color="b")])
    plot_trajectory_error(mu_hist, truth_hist, Sigma_hist)

