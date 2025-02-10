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
    np.array: Initial velocity assumed at the first timestep. [[v_x, v_y]].T
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

    return a_array, v_0
