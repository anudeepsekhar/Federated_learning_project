import os
import time
from multiprocessing import Process
from typing import Tuple

import flwr as fl
from matplotlib.pyplot import hist
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg, FedAdam
import dataset
import argparse
import pandas as pd

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

def start_client(dataset: DATASET, cid, writer, server_address="0.0.0.0", port=8080) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            """Fit model and return new weights as well as number of training
            examples."""
            model.set_weights(parameters)
            config['cid']=cid
            # print(config)
            history = model.fit(x_train, y_train, epochs=config['local_epochs'], batch_size=config['batch_size'], steps_per_epoch=config['steps_per_epoch'])
            data = history.history
            data['cid'] = cid
            data['round']=config['rnd']
            data['mode']=config['mode']
            writer.write(data)
            print('from client: ', data)

            return model.get_weights(), len(x_train), config

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            data = {}
            data['cid'] = cid
            data['round'] = config['rnd']
            data['mode'] = config['mode']
            data['loss'] = loss
            data['accuracy'] = accuracy
            writer.write(data)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(f'{server_address}:{port}', client=CifarClient())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--server_address', type=str, required=True)
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--port', type=str, required=False, default=8080)

    args = parser.parse_args()

    partitions = dataset.load(num_partitions=args.num_clients)

    dataset_ = partitions[args.client_id]

    start_client(dataset_)

