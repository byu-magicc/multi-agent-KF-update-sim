# Multi-agent KF update sim

This is a small python simulator for testing a potential method for using information from a factor graph multi-agent backend within an existing KF single-agent system.

To use, build the container, run the container, and copy out the results with the following:
```
podman build . -t multi-agent-gtsam-sim &&
podman run --name multi-agent-gtsam-sim_container --replace multi-agent-gtsam-sim &&
rm -rf plots &&
podman cp  multi-agent-gtsam-sim_container:/plots plots
```

To do Monte-Carlo simulations, add `-n 1000` or however many instances you want to run to the `podman run` command.
