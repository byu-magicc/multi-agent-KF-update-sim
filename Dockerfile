FROM docker.io/borglab/gtsam:4.2.0-tbb-ON-python-ON_22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install python3-numpy python3-matplotlib python3-tqdm -y

# Use if you want interactive plots. Requires giving container access to wayland or X11.
#RUN apt-get install python3-pyqt6.sip

COPY src /src

ENTRYPOINT [ "/src/main.py" ]
