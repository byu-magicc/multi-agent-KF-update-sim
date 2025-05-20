import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import numpy as np
from typing import List, Dict

# Detect if display backend is avaliable
if not os.environ.get('DISPLAY'):
    print('No display found. Using Agg backend for matplotlib.')
    matplotlib.use('Agg')


class Trajectory:
    """
    Struct for plot_overview function, containing all pose information and plotting parameters.
    """
    def __init__(self, pose_array: np.ndarray, name: str = "",
                 color: str = "b", opacity: float = 1.0):
        """
        pose_array: np.array (2, n)
            Array of all poses that a vehicle traveled during a trajectory.
        name: string
            Name to give trajectory in legend.
        color: string
            MatPlotLib color key, specifying the color to plot the trajectory.
        opacity: float, 0.0 <= opacity <=1.0
            Opacity to plot trajectory with.
        """
        assert pose_array.shape[0] == 2
        assert pose_array.ndim == 2
        assert opacity >= 0.0
        assert opacity <= 1.0

        self.poses = pose_array
        self.name = name
        self.color = color
        self.opacity = opacity


class Covariance:
    """
    Struct for plot_overview function, containing all Covariance ellipse information and plotting
    parameters.
    """
    def __init__(self, covariance_array: np.ndarray, mean: np.ndarray,
                 name: str = "", color:str = "b"):
        """
        covariance_array: np.array (2, 2)
            Covariance of ellipse.
        mean: np.array (2, 1)
            Location to center ellipse at.
        name: string
            Name to give ellipse in legend.
        color: string
            MatPlotLib color key, specifying the color to plot the ellipse.
        """
        assert covariance_array.shape == (2, 2)
        assert mean.shape == (2, 1)

        self.covariance_array = covariance_array
        self.mean = mean
        self.name = name
        self.color = color


class Markers:
    """
    Struct for plot_overview function, containing all marker information and plotting parameters.
    """
    def __init__(self, location: np.ndarray, color:str = "b"):
        """
        location: np.array (2, 1)
            Location to center marker at.
        color: string
            MatPlotLib color key, specifying the color to plot the marker.
        """
        assert location.shape == (2, 1)

        self.location = location
        self.color = color


class Lines:
    """
    Struct for plot_overview function, containing all line information and plotting parameters.
    """
    def __init__(self, endpoints: np.ndarray, color: str = "b"):
        """
        endpoints: np.array (2, 2)
            The start and end points of the line. [[x1, x2], [y1, y2]]
        color: string
            MatPlotLib color key, specifying the color to plot the line.
        """
        assert endpoints.shape == (2, 2)
        self.endpoints = endpoints
        self.color = color


def plot_overview(trajectories: List[Trajectory] = [],
                  covariances: List[Covariance] = [],
                  markers: List[Markers] = [],
                  lines: List[Lines] = [],
                  num_sigma: int = 2):
    """
    Plots the top-down view of various agent trajectories, covariance elipses, global measurement
    markers, and range markers.

    Parameters:
    trajectories: list of Trajectory classes
        Trajectories to plot.
    covariance: list of Covariance classes
        Covariance ellipses to plot.
    markers: list of Markers classes
        Global measurement markers to plot.
    lines: list of Lines classes
        Range measurements to plot as lines.
    num_sigma: int
        Number of standard deviations to plot for covariance ellipses.
    """
    assert num_sigma > 0

    plt.figure()

    # Plot range markers
    for line in lines:
        plt.plot(line.endpoints[0, :], line.endpoints[1, :], color=line.color, linewidth=0.75)

    # Plot trajectories
    for pose in trajectories:
        plt.plot(pose.poses[0, :], pose.poses[1, :], label=pose.name, color=pose.color,
                 alpha=pose.opacity)

    # Plot covariance ellipses
    for covariance in covariances:
        cov = covariance.covariance_array
        mean = covariance.mean.flatten()
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=num_sigma * np.sqrt(eigvals[0]) * 2,
                      height=num_sigma * np.sqrt(eigvals[1]) * 2,
                      angle=np.rad2deg(angle),
                      edgecolor=covariance.color,
                      facecolor="none",
                      zorder=4,
                      label=covariance.name
                      )
        plt.gca().add_artist(ell)

    # Plot global measurement markers
    for marker in markers:
        plt.plot(marker.location[0, :], marker.location[1, :], 'x', color=marker.color, markersize=8)

    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')

    if matplotlib.get_backend() == 'agg':
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/overview.svg')
        print('Saved overview.svg to file.')
    else:
        plt.show()


def plot_trajectory_error(time_hist: np.ndarray,
                          truth_hist: Dict[str, List[np.ndarray]],
                          ekf_mu_hist: Dict[str, List[np.ndarray]],
                          ekf_Sigma_hist: Dict[str, List[np.ndarray]],
                          backend_mu_hist: Dict[str, List[np.ndarray]] = {},
                          backend_Sigma_hist: Dict[str, List[np.ndarray]] = {},
                          plot_backend: bool = False,
                          num_sigma: int = 2,
                          sigma_only: bool = False):
    """
    Plots the individual components of the error in the trajectory estimate, with the covariance
    bound.

    Parameters:
    hist_indices: np.array of floats (n)
        Time of elements in history arrays.
    truth_hist: Dictionary of List of np.arrays (5 x n)
        Ground truth for every timestep for any number of iterations, where the key is the vehicle
        name.
    ekf_mu_hist: Dictionary of List of np.arrays (5 x n)
        State estimate of the EKF for every timestep for any number of iterations, where the key is
        the vehicle name.
    ekf_Sigma_hist: Dictionary of List of np.arrays (n x 5 x 5)
        Covariance of EKF estimate for every timestep for any number of iterations, where the key
        is the vehicle name.
    backend_mu_hist: Dictionary of List of np.arrays (3 x n)
        State estimate of the backend for every timestep for any number of iterations, where the key
        is the vehicle name. Ignored if plot_backend is False.
    backend_Sigma_hist: Dictionary of List of np.arrays (n x 3 x 3)
        Covariance of backend estimate for every timestep for any number of iterations, where the key
        is the vehicle name. Ignored if plot_backend is False.
    plot_backend: bool
        If True, plot the backend estimate as well.
    num_sigma: int
        Number of standard deviations to plot for covariance.
    sigma_only: bool
        If True, only plot the covariance bound. Useful for very large numbers of trajectories
    """

    # Check inputs
    for key in truth_hist.keys():
        assert key in ekf_mu_hist.keys()
        assert key in ekf_Sigma_hist.keys()
        assert len(truth_hist[key]) == len(ekf_mu_hist[key])
        assert len(truth_hist[key]) == len(ekf_Sigma_hist[key])
        if plot_backend:
            assert key in backend_mu_hist.keys()
            assert key in backend_Sigma_hist.keys()
            assert len(truth_hist[key]) == len(backend_mu_hist[key])
            assert len(truth_hist[key]) == len(backend_Sigma_hist[key])

        for i in range(len(truth_hist[key])):
            assert truth_hist[key][i].shape[0] == 5
            assert truth_hist[key][i].ndim == 2
            assert truth_hist[key][i].shape == ekf_mu_hist[key][i].shape
            assert truth_hist[key][i].shape[1] == ekf_Sigma_hist[key][i].shape[0]
            assert ekf_Sigma_hist[key][i].shape[1:] == (5, 5)
            if plot_backend:
                assert backend_mu_hist[key][i].shape[0] == 3
                assert truth_hist[key][i].shape[1] == backend_mu_hist[key][i].shape[1]
                assert truth_hist[key][i].shape[1] == backend_Sigma_hist[key][i].shape[0]
                assert backend_Sigma_hist[key][i].shape[1:] == (3, 3)

    # Check if we have multiple instances
    overlay_plots = len(next(iter(ekf_mu_hist.values()))) > 1
    alpha = 0.5 if overlay_plots else 1.0

    # Calculate population standard deviation
    ekf_error_sigma = {}
    ekf_error_nees = {}
    backend_error_sigma = {}
    backend_error_nees = {}
    for key in ekf_mu_hist.keys():
        ekf_residuals = []
        nees = []
        for i in range(len(ekf_mu_hist[key])):
            residuals = ekf_mu_hist[key][i] - truth_hist[key][i]
            ekf_residuals.append(residuals)

            Sigmas = ekf_Sigma_hist[key][i]
            nees_temp = []
            for j in range(residuals.shape[1]):
                res = residuals[:, j].reshape(-1, 1)
                Sigma = Sigmas[j]
                nees_temp.append((res.T @ np.linalg.inv(Sigma) @ res).item(0))
            nees.append(nees_temp)

        ekf_error_sigma[key] = np.sqrt(np.mean(np.array(ekf_residuals)**2, axis=0))
        ekf_error_nees[key] = np.mean(nees, axis=0)

        if plot_backend:
            backend_residuals = []
            nees = []
            for i in range(len(backend_mu_hist[key])):
                residuals = backend_mu_hist[key][i] - truth_hist[key][i][:3]
                backend_residuals.append(residuals)

                Sigmas = backend_Sigma_hist[key][i]
                nees_temp = []
                for j in range(residuals.shape[1]):
                    res = residuals[:, j].reshape(-1, 1)
                    Sigma = Sigmas[j]
                    nees_temp.append((res.T @ np.linalg.inv(Sigma) @ res).item(0))
                nees.append(nees_temp)

            backend_error_sigma[key] = np.sqrt(np.mean(np.array(backend_residuals)**2, axis=0))
            backend_error_nees[key] = np.mean(nees, axis=0)

    # Covariance only plots
    if sigma_only:
        # Error plots
        fig, axs = plt.subplots(6, len(ekf_mu_hist.keys()), figsize=(16, 12))
        if len(ekf_mu_hist.keys()) == 1:
            axs = np.expand_dims(axs, axis=1)
        column_idx = 0
        for key in ekf_mu_hist.keys():
            # X
            axs[0, column_idx].plot(time_hist, ekf_error_sigma[key][0, :],
                                    label='EKF', color='g')
            axs[0, column_idx].set_title(key)
            axs[0, column_idx].grid()
            if plot_backend:
                axs[0, column_idx].plot(time_hist, backend_error_sigma[key][0, :],
                                        label='FG', color='m')

            # Y
            axs[1, column_idx].plot(time_hist, ekf_error_sigma[key][1, :], color='g')
            axs[1, column_idx].grid()
            if plot_backend:
                axs[1, column_idx].plot(time_hist, backend_error_sigma[key][1, :], color='m')

            # Theta
            axs[2, column_idx].plot(time_hist, ekf_error_sigma[key][2, :], color='g')
            axs[2, column_idx].grid()
            if plot_backend:
                axs[2, column_idx].plot(time_hist, backend_error_sigma[key][2, :], color='m')

            # X Velocity
            axs[3, column_idx].plot(time_hist, ekf_error_sigma[key][3, :], color='g')
            axs[3, column_idx].grid()

            # Y Velocity
            axs[4, column_idx].plot(time_hist, ekf_error_sigma[key][4, :], color='g')
            axs[4, column_idx].grid()

            # NEES
            axs[5, column_idx].axhline(5.0, color='g', linestyle='--')
            axs[5, column_idx].plot(time_hist, ekf_error_nees[key], color='g')
            axs[5, column_idx].set_xlabel('Time (s)')
            axs[5, column_idx].grid()
            if plot_backend:
                axs[5, column_idx].axhline(3.0, color='m', linestyle='--')
                axs[5, column_idx].plot(time_hist, backend_error_nees[key], color='m')

            # Add ylabels and legend
            if column_idx == 0:
                axs[0, column_idx].set_ylabel('X Error (m)')
                axs[1, column_idx].set_ylabel('Y Error (m)')
                axs[2, column_idx].set_ylabel('Theta Error (rad)')
                axs[3, column_idx].set_ylabel('X Velocity Error (m/s)')
                axs[4, column_idx].set_ylabel('Y Velocity Error (m/s)')
                axs[5, column_idx].set_ylabel('NEES')
                axs[0, column_idx].legend()

            column_idx += 1

        plt.suptitle('Monte-Carlo Estimate Error')
        plt.tight_layout()

        if matplotlib.get_backend() == 'agg':
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/trajectory_error.svg')
            print('Saved trajectory_error.svg to file.')
        else:
            plt.show()

        return

    # Create full plot
    fig, axs = plt.subplots(5, len(ekf_mu_hist.keys()), figsize=(16, 12))
    if len(ekf_mu_hist.keys()) == 1:
        axs = np.expand_dims(axs, axis=1)

    column_idx = 0
    for key in ekf_mu_hist.keys():
        for i in range(len(ekf_mu_hist[key])):
            curr_truth = truth_hist[key][i]
            curr_ekf_mu = ekf_mu_hist[key][i]
            curr_ekf_sigma = np.sqrt(np.diagonal(ekf_Sigma_hist[key][i], axis1=1, axis2=2).T)
            ekf_error = curr_ekf_mu - curr_truth
            if plot_backend:
                curr_backend_mu = backend_mu_hist[key][i]
                curr_backend_sigma = np.sqrt(np.diagonal(backend_Sigma_hist[key][i], axis1=1, axis2=2).T)
                backend_error = curr_backend_mu - curr_truth[:3]

            # X position
            if i == 0:
                axs[0, column_idx].plot(time_hist, ekf_error[0, :], label='EKF Error',
                                        color='r', alpha=alpha)
                axs[0, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[0, :],
                                        label=f'EKF {num_sigma} Sigma', color='b')
            else:
                axs[0, column_idx].plot(time_hist, ekf_error[0, :], color='r', alpha=alpha)
                axs[0, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[0, :], color='b')
            axs[0, column_idx].plot(time_hist, -num_sigma*curr_ekf_sigma[0, :], color='b')

            if plot_backend:
                if i == 0:
                    axs[0, column_idx].plot(time_hist, backend_error[0, :], label='FG Error',
                                            color='y', alpha=alpha)
                    axs[0, column_idx].plot(time_hist, num_sigma*curr_backend_sigma[0, :],
                                            label=f'FG {num_sigma} Sigma', color='c')
                else:
                    axs[0, column_idx].plot(time_hist, backend_error[0, :], color='y', alpha=alpha)
                    axs[0, column_idx].plot(time_hist, num_sigma*curr_backend_sigma[0, :],
                                            color='c')
                axs[0, column_idx].plot(time_hist, -num_sigma*curr_backend_sigma[0, :], color='c')

            # Y position
            axs[1, column_idx].plot(time_hist, ekf_error[1, :], color='r', alpha=alpha)
            axs[1, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[1, :], color='b')
            axs[1, column_idx].plot(time_hist, -num_sigma*curr_ekf_sigma[1, :], color='b')

            if plot_backend:
                axs[1, column_idx].plot(time_hist, backend_error[1, :], color='y', alpha=alpha)
                axs[1, column_idx].plot(time_hist, num_sigma*curr_backend_sigma[1, :], color='c')
                axs[1, column_idx].plot(time_hist, -num_sigma*curr_backend_sigma[1, :], color='c')

            # Theta
            axs[2, column_idx].plot(time_hist, ekf_error[2, :], color='r', alpha=alpha)
            axs[2, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[2, :], color='b')
            axs[2, column_idx].plot(time_hist, -num_sigma*curr_ekf_sigma[2, :], color='b')

            if plot_backend:
                axs[2, column_idx].plot(time_hist, backend_error[2, :], color='y', alpha=alpha)
                axs[2, column_idx].plot(time_hist, num_sigma*curr_backend_sigma[2, :], color='c')
                axs[2, column_idx].plot(time_hist, -num_sigma*curr_backend_sigma[2, :], color='c')

            # X Velocity
            axs[3, column_idx].plot(time_hist, ekf_error[3, :], color='r', alpha=alpha)
            axs[3, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[3, :], color='b')
            axs[3, column_idx].plot(time_hist, -num_sigma*curr_ekf_sigma[3, :], color='b')

            # Y Velocity
            axs[4, column_idx].plot(time_hist, ekf_error[4, :], color='r', alpha=alpha)
            axs[4, column_idx].plot(time_hist, num_sigma*curr_ekf_sigma[4, :], color='b')
            axs[4, column_idx].plot(time_hist, -num_sigma*curr_ekf_sigma[4, :], color='b')

            # Formatting
            if i == 0:
                axs[0, column_idx].set_title(f'{key}')
                axs[0, column_idx].grid()
                axs[1, column_idx].grid()
                axs[2, column_idx].grid()
                axs[3, column_idx].grid()
                axs[4, column_idx].grid()
                axs[4, column_idx].set_xlabel('Time (s)')
            if column_idx == 0:
                axs[0, column_idx].set_ylabel('X Error (m)')
                axs[1, column_idx].set_ylabel('Y Error (m)')
                axs[2, column_idx].set_ylabel('Theta Error (rad)')
                axs[3, column_idx].set_ylabel('X Velocity Error (m/s)')
                axs[4, column_idx].set_ylabel('Y Velocity Error (m/s)')
                axs[0, column_idx].legend()

        column_idx += 1

    plt.suptitle('Estimate Error and Sigma')
    plt.tight_layout()

    if matplotlib.get_backend() == 'agg':
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/trajectory_error.svg')
        print('Saved trajectory_error.svg to file.')
    else:
        plt.show()


if __name__ == "__main__":
    from trajectories import sine_trajectory


    poses = [
        Trajectory(
            sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 100]]).T, 10, 5)[:2, :]
        ),
        Trajectory(
            sine_trajectory(100, np.array([[0, 0]]).T, np.array([[100, 0]]).T, 10, 5)[:2, :],
            name="Traj"
        )
    ]

    lines = [
        Lines(
            np.array([[0, 0], [100, 100]]).T
        ),
        Lines(
            np.array([[0, 0], [100, 0]]).T,
            color="r"
        )
    ]

    markers = [
        Markers(np.array([[50], [50]]), color="g"),
        Markers(np.array([[50], [50]]))
    ]

    covariances = [
        Covariance(
            np.array([[10, 0], [0, 1]]),
            np.array([[20], [50]]),
            color="b"
        ),
        Covariance(
            np.array([[12, -7], [-7, 6]]),
            np.array([[50], [20]]),
            color="k"
        )
    ]

    plot_overview(trajectories=poses, lines=lines, markers=markers, covariances=covariances)

    time_hist = np.array([0.0, 2.5])
    truth_hist = {"Vehicle 1": [np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).T,
                                np.array([[0, 0, 0, 0, 0], [-1, -1, -1, -1, -1]]).T]}
    ekf_mu_hist = {"Vehicle 1": [np.array([[0, 0, 0, 0, 0], [0.9, 0.9, 0.9, 0.9, 0.9]]).T,
                                 np.array([[0, 0, 0, 0, 0], [-0.9, -0.9, -0.9, -0.9, -0.9]]).T]}
    ekf_Sigma_hist = {"Vehicle 1": [np.array([np.eye(5)*0.1, np.eye(5)*0.2]),
                                    np.array([np.eye(5)*0.1, np.eye(5)*0.2])]}
    backend_mu_hist = {"Vehicle 1": [np.array([[0, 0, 0], [0.85, 0.85, 0.85]]).T,
                                     np.array([[0, 0, 0], [-0.85, -0.85, -0.85]]).T]}
    backend_Sigma_hist = {"Vehicle 1": [np.array([np.eye(3)*0.08, np.eye(3)*0.16]),
                                        np.array([np.eye(3)*0.08, np.eye(3)*0.16])]}

    plot_trajectory_error(time_hist, truth_hist, ekf_mu_hist, ekf_Sigma_hist,
                          backend_mu_hist, backend_Sigma_hist)
    plot_trajectory_error(time_hist, truth_hist, ekf_mu_hist, ekf_Sigma_hist,
                          backend_mu_hist, backend_Sigma_hist, plot_backend=True)
    plot_trajectory_error(time_hist, truth_hist, ekf_mu_hist, ekf_Sigma_hist,
                          backend_mu_hist, backend_Sigma_hist, sigma_only=True)
    plot_trajectory_error(time_hist, truth_hist, ekf_mu_hist, ekf_Sigma_hist,
                          backend_mu_hist, backend_Sigma_hist, plot_backend=True, sigma_only=True)

