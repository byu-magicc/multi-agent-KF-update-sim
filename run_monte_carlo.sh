#!/bin/bash

# This script is used for running very large Monte-Carlo simulations with all trajectory presets.
# Simulation is run headless in a podman container, useful for running on desktops over ssh.

# Clear previous plots, if they exist
rm -rf plots
mkdir plots

# Build podman container
podman build . -t multi-agent-gtsam-sim

for i in {0..5}; do
    echo "Running simulation $i..."

    # Run simulation
    podman run --name multi-agent-gtsam-sim_container --replace multi-agent-gtsam-sim --plot_fg_results -n 10000 -t $i

    # Copy out plots
    podman cp  multi-agent-gtsam-sim_container:/plots "plots/plots_$i"
done
