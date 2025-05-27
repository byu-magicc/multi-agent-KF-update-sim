import numpy as np


def _g(u_t, mu_t_1, dt):
    """
    IMU propagation model. Given the previous state and the current control input,
    return the predicted state.

    x_t = x_t-1 + v_t-1 * dt + 0.5 * R_theta @ a_t * dt^2
    theta_t = theta_t-1 + omega_t * dt
    v_t = v_t-1 + R_theta @ a_t * dt

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
        [[a_x, a_y, omega]].T
    mu_t_1: np.array, shape (5, 1)
        Previous state estimate. Contains the position, orientation, and velocity.
        [[x, y, theta, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    mu_t: np.array, shape (5, 1)
        Predicted state estimate.
    """

    a_t = u_t[0:2]
    omega_t = u_t[2]

    p_t_1 = mu_t_1[:2]
    theta_t_1 = mu_t_1[2]
    v_t_1 = mu_t_1[3:]

    R_theta = np.array([[np.cos(theta_t_1), -np.sin(theta_t_1)],
                       [np.sin(theta_t_1), np.cos(theta_t_1)]]).squeeze()

    p_t = p_t_1 + v_t_1 * dt + 0.5 * R_theta @ a_t * dt**2
    theta_t = theta_t_1 + omega_t * dt
    v_t = v_t_1 + R_theta @ a_t * dt

    return np.vstack([p_t, theta_t, v_t])


def _G_mu(u_t, mu_t_1, dt):
    """
    Jacobian of the IMU propagation model with respect to the state.

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
        [[a_x, a_y, omega]].T
    mu_t_1: np.array, shape (5, 1)
        State estimate at time t-1. Contains the position, orientation, and velocity.
        [[x, y, theta, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    G_mu_t: np.array, shape (5, 5)
    """

    a_t = u_t[0:2]
    theta_t_1 = mu_t_1[2]

    R_prime = np.array([[-np.sin(theta_t_1), -np.cos(theta_t_1)],
                        [np.cos(theta_t_1), -np.sin(theta_t_1)]]).squeeze()

    p_prime_theta = 0.5 * R_prime @ a_t * dt**2
    v_prime_theta = R_prime @ a_t * dt

    G_mu_t = np.array([[1, 0, p_prime_theta.item(0), dt, 0],
                       [0, 1, p_prime_theta.item(1), 0, dt],
                       [0, 0, 1, 0, 0],
                       [0, 0, v_prime_theta.item(0), 1, 0],
                       [0, 0, v_prime_theta.item(1), 0, 1]])

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
        [[x, y, theta, v_x, v_y]].T
    dt: float
        Time step.

    Returns:
    G_u_t: np.array, shape (5, 5)
    """

    theta_t_1 = mu_t_1[2]

    R_theta = np.array([[np.cos(theta_t_1), -np.sin(theta_t_1)],
                      [np.sin(theta_t_1), np.cos(theta_t_1)]]).squeeze()

    p_prime_a = 0.5 * R_theta * dt**2
    v_prime_a = R_theta * dt

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
        [[x, y, theta, v_x, v_y]].T

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
        [[x, y, theta, v_x, v_y]].T

    Returns:
    H_global_t: np.array, shape (3, 5)
        Jacobian of the global measurement model.
    """

    return np.eye(3, 5)


def _h_range(mu_a_t, mu_b_t):
    """
    Range measurement model.

    Parameters:
    mu_a_t: np.array, shape (5, 1)
        State estimate of vehicle a. [[x_a, y_a, theta_a, v_x_a, v_y_a]].T
    mu_b_t: np.array, shape (5, 1)
        State estimate of vehicle b. [[x_b, y_b, theta_b, v_x_b, v_y_b]].T
    """

    return np.linalg.norm(mu_a_t[:2] - mu_b_t[:2]).item(0)


def _H_range(mu_a_t, mu_b_t):
    """
    Jacobian of the range measurement model with respect to vehicle a's state.

    Parameters:
    mu_a_t: np.array, shape (5, 1)
        State estimate of vehicle a. [[x_a, y_a, theta_a, v_x_a, v_y_a]].T
    mu_b_t: np.array, shape (5, 1)
        State estimate of vehicle b. [[x_b, y_b, theta_b, v_x_b, v_y_b]].T
    """

    range = np.linalg.norm(mu_a_t[:2] - mu_b_t[:2]).item(0)

    x_a = mu_a_t.item(0)
    y_a = mu_a_t.item(1)
    x_b = mu_b_t.item(0)
    y_b = mu_b_t.item(1)

    return np.array([[(x_a - x_b) / range, (y_a - y_b) / range, 0, 0, 0]])


class EKF:
    """
    Basic EKF class for IMU propagation and global measurements.
    """
    def __init__(self, mu_0, Sigma_0, imu_sigmas, dt):
        """
        Parameters:
        mu_0: np.array, shape (5, 1)
            Initial state estimate. Contains the position, orientation, and velocity
            [[x, y, theta, v_x, v_y]].T
        Sigma_0: np.array, shape (5, 5)
            Initial covariance matrix.
        imu_sigmas: np.array, shape (3, 1)
            Standard deviations of the IMU noise.
            [[sigma_acc_x, sigma_acc_y, sigma_theta_dot]].T
        dt: float
            Time duration of IMU measurement steps, in seconds.
        """
        assert mu_0.shape == (5, 1)
        assert Sigma_0.shape == (5, 5)
        assert imu_sigmas.shape == (3, 1)

        self.mu = mu_0
        self.Sigma = Sigma_0
        self.R = np.diag(imu_sigmas.flatten()**2)
        self.dt = dt

        self.g = _g
        self.h_global = _h_global
        self.h_range = _h_range
        self.G_mu = _G_mu
        self.G_u = _G_u
        self.H_global = _H_global
        self.H_range = _H_range

    def propagate(self, u_t):
        """
        Propagate the state estimate and covariance matrix.

        Parameters:
        u_t: np.array, shape (3, 1)
            Control input at time t. Contains the acceleration and angular velocity.
            [[a_x, a_y, omega]].T
        """
        assert u_t.shape == (3, 1)

        # Calculate Jacobians
        G_mu_t = self.G_mu(u_t, self.mu, self.dt)
        G_u_t = self.G_u(u_t, self.mu, self.dt)

        self.mu = self.g(u_t, self.mu, self.dt)
        self.Sigma = G_mu_t @ self.Sigma @ G_mu_t.T + G_u_t @ self.R @ G_u_t.T

    def global_update(self, z_t, Q):
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
        self.Sigma = (np.eye(5) - K_t @ H_t) @ self.Sigma

    def range_update(self, z_t, Sigma_z, x_b, Sigma_b):
        """
        Apply a range measurement recieved between two vehicles. Correlation between vehicles is
        ignored, for simplicity.

        Parameters:
        z_t: float
            Range measurement.
        Q: float
            Variance of range measurement.
        x_b: np.array, shape (3, 1)
            Current state of other vehicle. [[x, y, theta, v_x, v_y]].T
        Sigma_b: np.array, shape (3, 3)
            Covariance of other vehicle state.
        """
        assert x_b.shape == (5, 1)
        assert Sigma_b.shape == (5, 5)

        H_t = self.H_range(self.mu, x_b)

        F = np.hstack((np.ones((1, 1)), -H_t, np.zeros((1, 2))))
        Sigma = np.zeros((6, 6))
        Sigma[0, 0] = Sigma_z
        Sigma[1:, 1:] = Sigma_b
        Q = F @ Sigma @ F.T

        K_t = self.Sigma @ H_t.T @ np.linalg.inv(H_t @ self.Sigma @ H_t.T + Q)
        self.mu = self.mu + K_t * (z_t - self.h_range(self.mu, x_b))
        self.Sigma = (np.eye(5) - K_t @ H_t) @ self.Sigma


if __name__ == "__main__":
    from measurements import get_imu_data
    from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
    from trajectories import sine_trajectory, line_trajectory

    np.set_printoptions(linewidth=np.inf)

    duration = 100
    dt = 1.0 / 400
    num_steps = int(duration / dt)
    imu_sigmas = np.array([[0.1, 0.1, np.deg2rad(5.)]]).T

    trajectory = sine_trajectory(num_steps, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 5, 2)
    time_hist = np.linspace(0, dt*(trajectory.shape[1] - 1), trajectory.shape[1])
    v_0 = ((trajectory[:2, 1] - trajectory[:2, 0]) / dt).reshape(-1, 1)

    imu_data, v_truth = get_imu_data(trajectory, imu_sigmas, v_0, dt)
    theta = trajectory[2, 0]

    mu_0 = np.vstack([trajectory[:, 0].reshape(-1, 1).copy(), v_0])
    Sigma_0 = np.eye(5) * 1e-15
    Sigma_global = np.diag([0.1, 0.1, 0.05])**2

    mu_hist = {"Vehicle 1": [mu_0]}
    truth_hist = {"Vehicle 1": [np.vstack((trajectory, v_truth))]}
    Sigma_hist = {"Vehicle 1": [Sigma_0.copy()]}

    ekf = EKF(mu_0, Sigma_0, imu_sigmas, dt)

    for i in range(num_steps - 1):
        ekf.propagate(imu_data[:, i].reshape(-1, 1))

        mu_hist["Vehicle 1"].append(ekf.mu.copy())
        Sigma_hist["Vehicle 1"].append(ekf.Sigma.copy())

        if i*dt % 50 == 0 and i != 0:
            global_meas = trajectory[:3, i + 1].reshape(-1, 1)
            ekf.global_update(global_meas, Sigma_global)

    mu_hist["Vehicle 1"] = [np.hstack(mu_hist["Vehicle 1"])]
    Sigma_hist["Vehicle 1"] = [np.array(Sigma_hist["Vehicle 1"])]

    plot_overview(trajectories=[Trajectory(trajectory[:2], name="Truth", color="r"),
                                Trajectory(mu_hist["Vehicle 1"][0][:2], name="Estimate", color="b")],
                  covariances=[Covariance(ekf.Sigma[:2, :2], ekf.mu[:2], color="b")])

    plot_trajectory_error(time_hist, truth_hist, mu_hist, Sigma_hist)

