#!/usr/bin/env python3

import argparse
import numpy as np
import time
from multiprocessing import Pool, cpu_count

from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
from simulator import Simulation


def run_simulation(_):
    return Simulation().run()


def main(num_instances: int):
    # Set random seed for reproducibility, but only if running a single instance
    # Multiple instances would return the same result if the same seed is used
    if num_instances == 1:
        np.random.seed(0)

    num_sigma = 2

    # Run simulations in parallel on multiple cores
    start_time = time.time()
    with Pool(processes=min(num_instances, cpu_count())) as pool:
        results = pool.map(run_simulation, range(num_instances))
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    num_vehicles = len(Simulation().vehicles)
    dt = Simulation().vehicles[0]._DT

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

        for i in range(num_vehicles):
            poses.append(Trajectory(truth_hist_array[i][:2, :], color="r"))

            if num_instances == 1:
                poses.append(Trajectory(mu_hist_array[i][:2, :], color="b"))
                covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                                              mu_hist_array[i][:2, -1].reshape(-1, 1),
                                              color="k"))
            else:
                poses.append(Trajectory(mu_hist_array[i][:2, :], color="b", opacity=0.5))
                if len(covariances) < num_vehicles:
                    covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                                                  truth_hist_array[i][:2, -1].reshape(-1, 1),
                                                  color="k"))
            if len(mu_hist) < num_vehicles:
                mu_hist[f'Vehicle {i}'] = [mu_hist_array[i]]
                truth_hist[f'Vehicle {i}'] = [truth_hist_array[i]]
                Sigma_hist[f'Vehicle {i}'] = [Sigma_hist_array[i]]
            else:
                mu_hist[f'Vehicle {i}'].append(mu_hist_array[i])
                truth_hist[f'Vehicle {i}'].append(truth_hist_array[i])
                Sigma_hist[f'Vehicle {i}'].append(Sigma_hist_array[i])

    poses[0].name = "Estimate"
    poses[1].name = "Truth"
    covariances[0].name = f"{num_sigma} Sigma Bound"

    if num_instances <= 100:
        plot_overview(poses, covariances, num_sigma=num_sigma)
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, dt, num_sigma=num_sigma)
    else:
        print('Only plotting sigma bounds since instances > 100')
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, dt, num_sigma=num_sigma, sigma_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-instances', type=int, default=1,
                        help='Number of instances to run simulation')
    args = vars(parser.parse_args())

    if args['num_instances'] < 1:
        raise ValueError("Number of instances must be at least 1")

    main(**args)
