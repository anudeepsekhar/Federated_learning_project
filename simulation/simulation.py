from asyncore import write
import os
import time
from multiprocessing import Process, Queue, SimpleQueue, Manager
from typing import Tuple
from csv_logger import Writer

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg, FedAdam
import dataset
from server import start_server
from client import start_client
from csv_logger import Writer
import pandas as pd
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_simulation(config):
    """Start a FL simulation."""
    # Load the dataset partitions
    partitions = dataset.load(num_partitions=config['num_clients'])
    writer = Writer()

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(config['num_rounds'], config['num_clients'], config['fraction_fit'])
    )
    server_process.start()
    processes.append(server_process)

    time.sleep(2)

    # Start all the clients
    for cid, partition in enumerate(partitions):
        client_process = Process(target=start_client, args=(partition,cid, writer))
        client_process.start()
        processes.append(client_process)
    

    # Block until all processes are finished
    for p in processes:        
        p.join()   

    writer.to_csv(os.path.join(config['exp_name'],'logs.csv'))

if __name__ == "__main__":
    config = {
        'exp_name':'exp2',
        'num_rounds':200,
        'num_clients':10,
        'fraction_fit':0.5

    }
    if not os.path.exists(config['exp_name']):
        os.makedirs(config['exp_name'])
    pd.DataFrame(config,index=[0]).to_csv(os.path.join(config['exp_name'],'config.csv'))
    run_simulation(config)
