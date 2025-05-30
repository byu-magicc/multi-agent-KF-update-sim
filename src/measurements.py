import numpy as np


def get_imu_data(trajectory, noise_std, v_0, dt):
    """
    Get simulated IMU data from the ground truth states.

    x_t = x_t-1 + v_t-1 * dt + 0.5 * a_t-1 * dt^2
    a_t = 2 * (x_t+1 - x_t - v_t * dt) / dt^2
    v_t = v_t-1 + a_t-1 * dt
    theta_dot_t = (theta_t+1 - theta_t) / dt

    Parameters:
    trajectory (np.array): 3xn Numpy array of full trajectory. [[x, y, theta], ...].T
    noise_std (np.array): Standard deviation of the noise for the IMU data. [[acc_x, acc_y, theta_dot]].T
    v_0 (np.array): 2x1 Numpy array of velocity at time 0. [[v_x, v_y]].T
    dt (float): Delta time step.

    Returns:
    np.array: IMU data at every timestep (minus the last). [[acc_x, acc_y, theta_dot], ...].T
    np.array: Initial forward velocity assumed at the first timestep.
    """
    assert trajectory.shape[0] == 3
    assert trajectory.ndim == 2
    assert trajectory.shape[1] > 1
    assert noise_std.shape == (3, 1)
    assert v_0.shape == (2, 1)
    assert dt > 0

    v_t = v_0.copy()
    a_array = []
    v_truth_array = [v_0]

    for t in range(1, trajectory.shape[1]):
        x_t1 = trajectory[:, t].reshape(-1, 1)
        x_t = trajectory[:, t - 1].reshape(-1, 1)

        # Calculate the acceleration in the global frame
        theta_dot = (x_t1[2] - x_t[2]) / dt
        a_t_global = 2 * (x_t1[:2] - x_t[:2] - v_t * dt) / dt ** 2
        # Timestep here
        v_t += a_t_global[:2] * dt

        # Rotate the acceleration to the body frame
        rot = np.array([[np.cos(x_t[2]), -np.sin(x_t[2])], [np.sin(x_t[2]), np.cos(x_t[2])]]).squeeze()
        a_t_body = rot.T @ a_t_global

        a_array.append(np.array([a_t_body[0], a_t_body[1], theta_dot]))
        v_truth_array.append(v_t.copy())

    # Add noise
    a_array = np.hstack(a_array)
    a_array += np.random.normal(0, noise_std, a_array.shape)

    return a_array, np.hstack(v_truth_array)


def get_pseudo_measurement(mu_current, mu_desired, Sigma_current, Sigma_desired):
    """
    Get pseudo measurement. Pseudo measurements are measurements intended to force an estimator
    from one mean and covariance to another mean and covariance. Can use measurement type, so long
    as the jacobian relating the measurement to the state is the identity matrix.

    Parameters:
    mu_current (np.array): nx1 numpy array of current filter state.
    mu_desired (np.array): nx1 numpy array of desired filter state.
    Sigma_current (np.array): nxn numpy array of current filter covariance.
    Sigma_desired (np.array): nxn numpy array of desired filter covariance.

    Returns:
    z, Sigma_z: Pseudo global measurement and covariance.
    """
    dim = mu_current.shape[0]
    assert mu_current.shape == (dim, 1)
    assert mu_desired.shape == (dim, 1)
    assert Sigma_current.shape == (dim, dim)
    assert Sigma_desired.shape == (dim, dim)

    temp = np.linalg.inv(np.eye(dim) - Sigma_desired @ np.linalg.inv(Sigma_current))
    Sigma_z = (temp - np.eye(dim)) @ Sigma_current
    z = temp @ (mu_desired - mu_current) + mu_current

    return z, Sigma_z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotters import plot_overview, Trajectory
    from trajectories import line_trajectory, arc_trajectory, sine_trajectory

    duration = 100
    dt = 1.0 / 100
    num_steps = int(duration / dt)
    noise_std = np.array([[0., 0., np.deg2rad(0.)]]).T

    x = np.array([[0, 0]], dtype=float).T

    trajectory = line_trajectory(num_steps, x, np.array([[10, 10]]).T)
    trajectory = arc_trajectory(num_steps, x, np.array([[10, 0]]).T, np.deg2rad(15))
    trajectory = sine_trajectory(num_steps, x, np.array([[100, 100]]).T, 5, 2)

    v_0 = ((trajectory[:2, 1] - trajectory[:2, 0]) / dt).reshape(-1, 1)

    imu_data, _ = get_imu_data(trajectory, noise_std, v_0, dt)

    x = trajectory[:, 0].reshape(-1, 1).copy()
    x_traj = [x.copy()]

    v = v_0.copy()

    for i in range(num_steps - 1):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                      [np.sin(x[2]), np.cos(x[2])]]).squeeze()

        x[:2] += v * dt + 0.5 * R @ imu_data[:2, i].reshape(-1, 1) * dt ** 2
        x[2] += imu_data[2, i] * dt
        v += R @ imu_data[:2, i].reshape(-1, 1) * dt

        x_traj.append(x.copy())

        #print(f"Step {i}: x = {x.T}, v = {v.T}, imu = {imu_data[:, i]}")

    x_traj = np.hstack(x_traj)

    plot_overview([Trajectory(trajectory[:2], "Truth", "r"), Trajectory(x_traj[:2], "Estimate", "b")])

    error = x_traj[:2] - trajectory[:2]
    error_theta = x_traj[2] - trajectory[2]
    plt.plot(error[0, :], label="x error")
    plt.plot(error[1, :], label="y error")
    plt.plot(error_theta, label="theta error")
    plt.legend()
    plt.show()

