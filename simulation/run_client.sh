#!/bin/bash
set -e
sudo docker build -t fedl:latest . && \
sudo docker run --name fedlsim_$CLIENT_ID -it  --network host --rm -v $PWD:/workspace -w /workspace fedl:latest python client.py --server_address $SERVER_ADDRESS --num_clients 10 --client_id $CLIENT_ID 
