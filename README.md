# Multi-agent KF update sim

This is a small python simulator for testing a potential method for using information from a factor graph multi-agent backend within an existing KF single-agent system.

Tested with GTSAM docker image (in Distrobox): docker.io/borglab/gtsam:4.2.0-tbb-ON-python-ON_22.04

Install apt dependencies and run code (I'm not using virtual environments as system installations of GTSAM sometimes struggled with those for some reason):
```
sudo apt install python3-numpy python3-matplotlib python3-pyqt6.sip
```
