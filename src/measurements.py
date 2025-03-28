import numpy as np


def get_odom_data(trajectory, alphas=None):
    """
    Get simulated odometry from the ground truth states.

    Uses rotate, translate, rotate odometry, based on section 5.4 in Probabilistic Robotics.
    rot_1 = arctan2(y_t - y_t_1, x_t - x_t_1) - psi_t_1
    trans = sqrt((x_t - x_t_1)^2 + (y_t - y_t_1)^2)
    rot_2 = psi_t - psi_t_1 - rot_1

    Parameters:
    trajectory (np.array): 3xn Numpy array of full trajectory. [[x, y, psi], ...].T
    alphas (np.array): alpha values for noise, see table 5.6 in Probabilistic Robotics.
        Default is None, which means no noise is added.
        [a_1, a_2, a_3, a_4]

    Returns:
    np.array: Odometry data at every timestep (excluding the first). [[rot_1, trans, rot_2], ...].T
    """
    assert trajectory.shape[0] == 3
    assert trajectory.ndim == 2
    assert trajectory.shape[1] > 1

    odom_measurements = []
    for t in range(1, trajectory.shape[1]):
        rot_1 = np.arctan2(trajectory[1, t] - trajectory[1, t - 1],
                           trajectory[0, t] - trajectory[0, t - 1]) - trajectory[2, t - 1]
        trans = np.sqrt((trajectory[0, t] - trajectory[0, t - 1]) ** 2 \
            + (trajectory[1, t] - trajectory[1, t - 1]) ** 2)
        rot_2 = trajectory[2, t] - trajectory[2, t - 1] - rot_1

        if alphas is not None:
            odom = np.zeros((3,1))
            odom[0] = np.random.normal(rot_1, np.sqrt(alphas[0]*rot_1**2 + alphas[1]*trans**2))
            odom[1] = np.random.normal(trans, np.sqrt(alphas[2]*trans**2 + alphas[3]*(rot_1**2 + rot_2**2)))
            odom[2] = np.random.normal(rot_2, np.sqrt(alphas[0]*rot_2**2 + alphas[1]*trans**2))
            odom_measurements.append(odom)
        else:
            odom_measurements.append(np.array([rot_1, trans, rot_2]).reshape(-1, 1))

    odom_measurements = np.hstack(odom_measurements)

    return odom_measurements


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotters import plot_overview, Trajectory
    from trajectories import line_trajectory, arc_trajectory, sine_trajectory

    num_steps = 100
    alphas = np.array([0.25, 0.005, 0.25, 0.05])**2

    x = np.array([[0, 0]], dtype=float).T

    trajectory = line_trajectory(num_steps, x, np.array([[10, 10]]).T)
    trajectory = arc_trajectory(num_steps, x, np.array([[10, 0]]).T, 10)
    trajectory = sine_trajectory(num_steps, x, np.array([[100, 100]]).T, 5, 2)
    odom_data = get_odom_data(trajectory, alphas)

    x_traj = [trajectory[:, 0].reshape(-1, 1).copy()]
    x = trajectory[:, 0].reshape(-1, 1).copy()

    for i in range(num_steps - 1):

        x[0] += odom_data[1, i] * np.cos(x[2] + odom_data[0, i])
        x[1] += odom_data[1, i] * np.sin(x[2] + odom_data[0, i])
        x[2] += odom_data[0, i] + odom_data[2, i]

        x_traj.append(x.copy())

        print(f"Step {i}: x = {x.T}, odom = {odom_data[:, i]}")

    x_traj = np.hstack(x_traj)

    plot_overview([Trajectory(trajectory[:2], "Truth", "r"), Trajectory(x_traj[:2], "Estimate", "b")])

    error = x_traj[:2] - trajectory[:2]
    error_psi = x_traj[2] - trajectory[2]
    plt.plot(error[0, :], label="x error")
    plt.plot(error[1, :], label="y error")
    plt.plot(error_psi, label="psi error")
    plt.legend()
    plt.show()
