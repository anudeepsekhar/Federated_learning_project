#!/bin/bash
set -e
sudo docker build -t fedl:latest . && \
sudo docker run --name fedlsim -it -p --network host --rm -v $PWD:/workspace -w /workspace fedl:latest python server.py --num_rounds 100 --num_clients 10 --fraction_fit 0.5
