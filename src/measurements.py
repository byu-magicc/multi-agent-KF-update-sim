import numpy as np


def get_odom_data(trajectory, odom_sigmas):
    """
    Get simulated odometry from the ground truth states.

    Parameters:
    trajectory (np.array): 3xn Numpy array of full trajectory. [[x, y, theta], ...].T
    odom_sigmas (np.array): 3x1 Numpy array of odometry noise standard deviations.
        [[sigma_x, sigma_y, sigma_theta]].T

    Returns:
    np.array: Odometry data at every timestep (excluding the first). [[delta_x, delta_y, delta_theta]], ...].T
    """
    assert trajectory.shape[0] == 3
    assert trajectory.ndim == 2
    assert trajectory.shape[1] > 1

    odom_measurements = []
    for t in range(1, trajectory.shape[1]):
        R = np.array([[np.cos(trajectory[2, t-1]), -np.sin(trajectory[2, t-1])],
                      [np.sin(trajectory[2, t-1]), np.cos(trajectory[2, t-1])]])
        delta_xy = R.T @ (trajectory[:2, t] - trajectory[:2, t-1]).reshape(-1, 1)
        odom = np.vstack([delta_xy, trajectory[2, t] - trajectory[2, t-1]])
        odom += np.random.normal(0, odom_sigmas)
        odom_measurements.append(odom)
    odom_measurements = np.hstack(odom_measurements)

    return odom_measurements


def get_pseudo_global_measurement(mu_current, mu_desired, Sigma_current, Sigma_desired):
    """
    Get pseudo global measurement. Pseudo global measurements are measurements intended to
    force an estimator from one mean and covariance to another mean and covariance.

    Parameters:
    mu_current (np.array): 3x1 numpy array of current filter state. [[x, y, theta]].T
    mu_desired (np.array): 3x1 numpy array of desired filter state. [[x, y, theta]].T
    Sigma_current (np.array): 3x3 numpy array of current filter covariance.
    Sigma_desired (np.array): 3x3 numpy array of desired filter covariance.

    Returns:
    z, Sigma_z: Pseudo global measurement and covariance.
    """
    assert mu_current.shape == (3, 1)
    assert mu_desired.shape == (3, 1)
    assert Sigma_current.shape == (3, 3)
    assert Sigma_desired.shape == (3, 3)

    temp = np.linalg.inv(np.eye(3) - Sigma_desired @ np.linalg.inv(Sigma_current))
    Sigma_z = (temp - np.eye(3)) @ Sigma_current
    z = temp @ (mu_desired - mu_current) + mu_current

    return z, Sigma_z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotters import plot_overview, Trajectory
    from trajectories import line_trajectory, arc_trajectory, sine_trajectory

    num_steps = 100
    odom_sigmas = np.array([0.1, 0.1, np.deg2rad(1)]).reshape(-1, 1)

    x = np.array([[0, 0]], dtype=float).T

    trajectory = line_trajectory(num_steps, x, np.array([[10, 10]]).T)
    trajectory = arc_trajectory(num_steps, x, np.array([[10, 0]]).T, np.deg2rad(15))
    trajectory = sine_trajectory(num_steps, x, np.array([[100, 100]]).T, 5, 2)
    odom_data = get_odom_data(trajectory, odom_sigmas)

    x = trajectory[:, 0].reshape(-1, 1).copy()
    x_traj = [x.copy()]

    for i in range(num_steps - 1):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                      [np.sin(x[2]), np.cos(x[2])]]).squeeze()
        x[:2] += R @ odom_data[:2, i].reshape(-1, 1)
        x[2] += odom_data[2, i]

        x_traj.append(x.copy())

        print(f"Step {i}: x = {x.T}, odom = {odom_data[:, i]}")

    x_traj = np.hstack(x_traj)

    plot_overview([Trajectory(trajectory[:2], "Truth", "r"), Trajectory(x_traj[:2], "Estimate", "b")])

    error = x_traj[:2] - trajectory[:2]
    error_theta = x_traj[2] - trajectory[2]
    plt.plot(error[0, :], label="x error")
    plt.plot(error[1, :], label="y error")
    plt.plot(error_theta, label="theta error")
    plt.legend()
    plt.show()
