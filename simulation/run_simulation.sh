#!/bin/bash
set -e
docker build -t fedl:latest . && \
docker run --name fedlsim -it -p 8080:8080 --rm -v $PWD:/workspace -w /workspace fedl:latest python simulation.py
