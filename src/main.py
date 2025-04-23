#!/usr/bin/env python3

import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
from simulator import Simulation


def run_simulation(args):
    thread_id, num_steps_in_results, plot_fg_results = args
    np.random.seed(thread_id)
    return Simulation().run(num_steps_in_results, plot_fg_results)


def main(num_instances: int, plot_fg_results: bool):
    num_sigma = 2
    large_iteration_cutoff = 10

    # Run simulations in parallel on multiple cores
    num_steps_in_results = 25 if plot_fg_results else 100
    with Pool(processes=min(num_instances, int(cpu_count() / 2))) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(run_simulation,
                                [(i, num_steps_in_results, plot_fg_results)
                                 for i in range(num_instances)]),
            total=num_instances,
            desc="Simulating"
        ):
            results.append(result)

    num_vehicles = len(Simulation().vehicles)

    # Extract data for plotting
    poses = []
    covariances = []
    hist_indices = results[0][0]
    truth_hist = {}
    ekf_mu_hist = {}
    ekf_Sigma_hist = {}
    backend_mu_hist = {}
    backend_Sigma_hist = {}
    for result in results:
        truth_hist_array = result[1]
        ekf_hist_mu_array = result[2]
        ekf_hist_Sigma_array = result[3]
        backend_hist_mu_array = result[4]
        backend_hist_Sigma_array = result[5]

        for i in range(num_vehicles):
            poses.append(Trajectory(ekf_hist_mu_array[i][:2, :], color="b", opacity=0.5))
            if plot_fg_results:
                poses.append(Trajectory(backend_hist_mu_array[i][:2, :], color="g", opacity=0.5))

            if len(ekf_mu_hist) < num_vehicles:  # First iteration
                poses.append(Trajectory(truth_hist_array[i][:2, :], color="r"))
                truth_hist[f'Vehicle {i}'] = [truth_hist_array[i]]
                ekf_mu_hist[f'Vehicle {i}'] = [ekf_hist_mu_array[i]]
                ekf_Sigma_hist[f'Vehicle {i}'] = [ekf_hist_Sigma_array[i]]
                covariances.append(Covariance(ekf_hist_Sigma_array[i][-1, :2, :2],
                                              truth_hist_array[i][:2, -1].reshape(-1, 1),
                                              color="k"))
                if plot_fg_results:
                    backend_mu_hist[f'Vehicle {i}'] = [backend_hist_mu_array[i]]
                    backend_Sigma_hist[f'Vehicle {i}'] = [backend_hist_Sigma_array[i]]
                    covariances.append(Covariance(backend_hist_Sigma_array[i][-1, :2, :2],
                                                  truth_hist_array[i][:2, -1].reshape(-1, 1),
                                                  color="g"))
            else:
                truth_hist[f'Vehicle {i}'].append(truth_hist_array[i])
                ekf_mu_hist[f'Vehicle {i}'].append(ekf_hist_mu_array[i])
                ekf_Sigma_hist[f'Vehicle {i}'].append(ekf_hist_Sigma_array[i])

                if plot_fg_results:
                    backend_mu_hist[f'Vehicle {i}'].append(backend_hist_mu_array[i])
                    backend_Sigma_hist[f'Vehicle {i}'].append(backend_hist_Sigma_array[i])

    poses[0].name = "EKF"
    if plot_fg_results:
        poses[1].name = "FG"
        poses[2].name = "Truth"
    else:
        poses[1].name = "Truth"
    covariances[0].name = f"EKF {num_sigma} Sigma"
    if plot_fg_results:
        covariances[1].name = f"FG {num_sigma} Sigma"

    if num_instances <= large_iteration_cutoff:
        plot_overview(poses, covariances, num_sigma=num_sigma)
        plot_trajectory_error(hist_indices, truth_hist, ekf_mu_hist, ekf_Sigma_hist, backend_mu_hist,
                              backend_Sigma_hist, plot_backend=plot_fg_results)
    else:
        print(f'Only plotting sigma bounds since instances > {large_iteration_cutoff}')
        plot_trajectory_error(hist_indices, truth_hist, ekf_mu_hist, ekf_Sigma_hist, backend_mu_hist,
                              backend_Sigma_hist, plot_backend=plot_fg_results, sigma_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-instances', type=int, default=1,
                        help='Number of instances to run simulation')
    parser.add_argument('--plot_fg_results', action='store_true',
                        help='Plot consistency results for the factor grath. ' + \
                        'Slows simulation significantly.')
    args = vars(parser.parse_args())

    if args['num_instances'] < 1:
        raise ValueError("Number of instances must be at least 1")

    main(**args)
