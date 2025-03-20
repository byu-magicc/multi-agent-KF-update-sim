import numpy as np


def get_imu_data(trajectory, noise_std, dt):
    """
    Get simulated IMU data from the ground truth states.

    x_t = x_t-1 + v_t-1 * dt + 0.5 * a_t-1 * dt^2
    a_t = 2 * (x_t+1 - x_t - v_t * dt) / dt^2
    v_t = v_t-1 + a_t-1 * dt
    psi_dot_t = (psi_t+1 - psi_t) / dt

    Parameters:
    trajectory (np.array): 3xn Numpy array of full trajectory. [[x, y, psi], ...].T
    noise_std (np.array): Standard deviation of the noise for the IMU data. [[acc_x, acc_y, psi_dot]].T
    dt (float): Delta time step.

    Returns:
    np.array: IMU data at every timestep (minus the last). [[acc_x, acc_y, psi_dot], ...].T
    np.array: Initial forward velocity assumed at the first timestep.
    """
    assert trajectory.shape[0] == 3
    assert trajectory.ndim == 2
    assert trajectory.shape[1] > 1
    assert noise_std.shape == (3, 1)
    assert dt > 0

    # Initialize the array with zero acceleration in first time step
    v_0 = ((trajectory[:2, 1] - trajectory[:2, 0]) / dt).reshape(-1, 1)
    psi_dot_0 = (trajectory[2, 1] - trajectory[2, 0]) / dt
    v_t = v_0.copy()
    a_array = [np.array([[0, 0, psi_dot_0]], dtype=float).T]

    for t in range(2, trajectory.shape[1]):
        x_t1 = trajectory[:, t].reshape(-1, 1)
        x_t = trajectory[:, t - 1].reshape(-1, 1)

        # Calculate the acceleration in the global frame
        psi_dot = (x_t1[2] - x_t[2]) / dt
        a_t_global = 2 * (x_t1[:2] - x_t[:2] - v_t * dt) / dt ** 2
        # Timestep here
        v_t += a_t_global[:2] * dt

        # Rotate the acceleration to the body frame
        rot = np.array([[np.cos(x_t[2]), -np.sin(x_t[2])], [np.sin(x_t[2]), np.cos(x_t[2])]]).squeeze()
        a_t_body = rot.T @ a_t_global

        a_array.append(np.array([a_t_body[0], a_t_body[1], psi_dot]))

    # Add noise
    a_array = np.hstack(a_array)
    a_array += np.random.normal(0, noise_std, a_array.shape)

    return a_array, np.linalg.norm(v_0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotters import plot_overview
    from trajectories import line_trajectory, arc_trajectory, sine_trajectory

    time = 60
    dt = 1.0 / 400
    num_steps = int(time / dt)
    noise_std = np.array([[0., 0., 0.]]).T

    x = np.array([[0, 0]], dtype=float).T

    trajectory = line_trajectory(num_steps, x, np.array([[10, 10]]).T)
    trajectory = arc_trajectory(num_steps, x, np.array([[10, 0]]).T, 10)
    trajectory = sine_trajectory(num_steps, x, np.array([[100, 100]]).T, 5, 2)
    imu_data, v = get_imu_data(trajectory, noise_std, dt)

    x_traj = [trajectory[:, 0].reshape(-1, 1).copy()]
    x = trajectory[:, 0].reshape(-1, 1).copy()

    for i in range(num_steps - 1):
        rot = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]]).squeeze()

        x[:2] += v * dt + 0.5 * rot @ imu_data[:2, i].reshape(-1, 1) * dt ** 2
        x[2] += imu_data[2, i] * dt
        v += rot @ imu_data[:2, i].reshape(-1, 1) * dt

        x_traj.append(x.copy())

        print(f"Step {i}: x = {x.T}, v = {v.T}, imu = {imu_data[:, i]}")

    x_traj = np.hstack(x_traj)

    plot_overview([[trajectory[:2], "Truth", "r"], [x_traj[:2], "Estimate", "b"]])

    error = x_traj[:2] - trajectory[:2]
    error_psi = x_traj[2] - trajectory[2]
    plt.plot(error[0, :], label="x error")
    plt.plot(error[1, :], label="y error")
    plt.plot(error_psi, label="psi error")
    plt.legend()
    plt.show()
