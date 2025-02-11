import numpy as np
import jax.numpy as jnp
from jax import jacrev, jit
import jax

jax.config.update("jax_enable_x64", True)


@jit
def _g(u_t, mu_t_1, dt):
    """
    IMU measurement propagation model. Given the previous state and the current control input,
    return the predicted state.

    x_t = x_t-1 + v_t-1 * dt + 0.5 * R_psi @ a_t * dt^2
    psi_t = psi_t-1 + omega_t * dt
    v_t = v_t-1 + R_psi @ a_t * dt

    Parameters:
    u_t: np.array, shape (3, 1)
        Control input at time t. Contains the acceleration and angular velocity.
    mu_t_1: np.array, shape (5, 1)
        Previous state estimate. Contains the position, orientation, and velocity.
    dt: float
        Time step.

    returns:
    mu_t: np.array, shape (5, 1)
        Predicted state estimate.
    """

    a_t = u_t[0:2]
    omega_t = u_t[2]

    p_t_1 = mu_t_1[:2]
    psi_t_1 = mu_t_1[2]
    v_t_1 = mu_t_1[3:]

    R_psi = jnp.array([[jnp.cos(psi_t_1), -jnp.sin(psi_t_1)],
                       [jnp.sin(psi_t_1), jnp.cos(psi_t_1)]]).squeeze()

    p_t = p_t_1 + v_t_1 * dt + 0.5 * R_psi @ a_t * dt**2
    psi_t = psi_t_1 + omega_t * dt
    v_t = v_t_1 + R_psi @ a_t * dt

    return jnp.vstack([p_t, psi_t, v_t])


@jit
def _h_global(mu_t):
    """
    Global measurement model. Given the current state, return the predicted global position.
    Currently, the measurement is the same as the state of the position.

    Parameters:
    mu_t: np.array, shape (5, 1)
        Current state estimate. Contains the position, orientation, and velocity.

    Returns:
    p_t: np.array, shape (2, 1)
        Predicted global position.
    """

    p_t = mu_t[:2]

    return p_t


class EKF:
    def __init__(self, mu_0, Sigma_0, Sigma_global, Sigma_imu):
        assert mu_0.shape == (5, 1)
        assert Sigma_0.shape == (5, 5)
        assert Sigma_global.shape == (2, 2)
        assert Sigma_imu.shape == (3, 3)

        self.mu = mu_0
        self.Sigma = Sigma_0

        self.Q = Sigma_global
        self.R = Sigma_imu

        self.g = _g
        self.h = _h_global
        self.G = jacrev(self.g, argnums=1)
        self.H_imu = jacrev(self.g, argnums=0)
        self.H_global = jacrev(self.h, argnums=0)

    def propagate(self, u_t, dt):
        G_t = self.G(u_t, self.mu, dt).squeeze()
        H_t = self.H_imu(u_t, self.mu, dt).squeeze()

        self.mu = self.g(u_t, self.mu, dt)
        self.Sigma = G_t @ self.Sigma @ G_t.T + H_t @ self.R @ H_t.T

    def update_global(self, z_t):
        H_t = self.H_global(self.mu)

        K_t = self.Sigma @ H_t.T @ np.linalg.inv(H_t @ self.Sigma @ H_t.T + self.Q)
        self.mu = self.mu + K_t @ (z_t - self.h(self.mu))
        self.Sigma = (np.eye(5) - K_t @ H_t) @ self.Sigma


if __name__ == "__main__":
    from measurements import get_imu_data
    from plotters import plot_overview
    from trajectories import sine_trajectory

    import matplotlib.pyplot as plt

    import time

    np.set_printoptions(linewidth=np.inf)

    total_time = 60
    dt = 1.0 / 200
    num_steps = int(total_time / dt)
    imu_noise_std = np.array([[0.1, 0.1, 0.05]]).T

    trajectory = sine_trajectory(num_steps, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 5, 2)

    imu_data, v_0 = get_imu_data(trajectory, imu_noise_std, dt)

    mu_0 = np.vstack([trajectory[:, 0].reshape(-1, 1).copy(), v_0])
    Sigma_0 = np.eye(5) * 1e-9
    Sigma_global = np.eye(2) * 1e-9
    Sigma_imu = np.diag(imu_noise_std.squeeze()**2)
    mu_hist = [mu_0]

    init_time = time.time()
    ekf = EKF(mu_0, Sigma_0, Sigma_global, Sigma_imu)

    for i in range(num_steps - 1):
        if i % 1000 == 0:
            print(f"Percent {i / num_steps * 100:.1f}%")

        ekf.propagate(imu_data[:, i].reshape(-1, 1), dt)
        mu_hist.append(ekf.mu)

    print(f"Time taken: {time.time() - init_time:.1f}s")

    mu_hist = np.hstack(mu_hist)

    plot_overview(poses=[[trajectory[:2], "Truth", "r"], [mu_hist[:2], "Estimate", "b"]],
                         covariances=[[ekf.Sigma[:2, :2], ekf.mu[:2], "b"]])

    error = mu_hist[:2] - trajectory[:2]
    error_psi = mu_hist[2] - trajectory[2]
    plt.plot(error[0, :], label="x error")
    plt.plot(error[1, :], label="y error")
    plt.plot(error_psi, label="psi error")
    plt.legend()
    plt.show()
