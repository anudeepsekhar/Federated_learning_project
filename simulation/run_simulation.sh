#!/bin/bash
set -e
sudo docker build -t fedl:latest . && \
sudo docker run --name fedlsim -it --network host --rm -v $PWD:/workspace -w /workspace fedl:latest python simulation.py
