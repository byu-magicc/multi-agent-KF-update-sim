#!/usr/bin/env python3

import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
from simulator import Simulation


def run_simulation(args):
    thread_id, compress_results = args
    np.random.seed(thread_id)
    return Simulation().run(compress_results=compress_results)


def main(num_instances: int):
    num_sigma = 2
    large_iteration_cutoff = 50

    # Run simulations in parallel on multiple cores
    compress_results = True if num_instances > large_iteration_cutoff else False
    with Pool(processes=min(num_instances, int(cpu_count() / 2))) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(run_simulation,
                                [(i, compress_results) for i in range(num_instances)]),
            total=num_instances,
            desc="Simulating"
        ):
            results.append(result)

    num_vehicles = len(Simulation().vehicles)

    # Extract data for plotting
    poses = []
    covariances = []
    mu_hist = {}
    truth_hist = {}
    Sigma_hist = {}
    for result in results:
        mu_hist_array = result[0]
        truth_hist_array = result[1]
        Sigma_hist_array = result[2]
        backend_mu_hist_array = result[3]
        backend_Sigma_hist_array = result[4]

        for i in range(num_vehicles):
            poses.append(Trajectory(mu_hist_array[i][:2, :], color="b", opacity=0.5))
            poses.append(Trajectory(backend_mu_hist_array[i][:2, :], color="g", opacity=0.5))

            if len(mu_hist) < num_vehicles:  # First iteration
                poses.append(Trajectory(truth_hist_array[i][:2, :], color="r"))
                mu_hist[f'Vehicle {i}'] = [mu_hist_array[i]]
                truth_hist[f'Vehicle {i}'] = [truth_hist_array[i]]
                Sigma_hist[f'Vehicle {i}'] = [Sigma_hist_array[i]]
                covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                                              truth_hist_array[i][:2, -1].reshape(-1, 1),
                                              color="k"))
                covariances.append(Covariance(backend_Sigma_hist_array[i][-1, :2, :2],
                                              truth_hist_array[i][:2, -1].reshape(-1, 1),
                                              color="g"))
            else:
                mu_hist[f'Vehicle {i}'].append(mu_hist_array[i])
                truth_hist[f'Vehicle {i}'].append(truth_hist_array[i])
                Sigma_hist[f'Vehicle {i}'].append(Sigma_hist_array[i])

    poses[0].name = "EKF"
    poses[1].name = "FG"
    poses[2].name = "Truth"
    covariances[0].name = f"EKF {num_sigma} Sigma"
    covariances[1].name = f"FG {num_sigma} Sigma"

    if num_instances <= large_iteration_cutoff:
        plot_overview(poses, covariances, num_sigma=num_sigma)
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, num_sigma=num_sigma)
    else:
        print('Only plotting sigma bounds since instances > 100')
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, num_sigma=num_sigma, sigma_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-instances', type=int, default=1,
                        help='Number of instances to run simulation')
    args = vars(parser.parse_args())

    if args['num_instances'] < 1:
        raise ValueError("Number of instances must be at least 1")

    main(**args)
