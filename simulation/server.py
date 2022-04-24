import os
from xmlrpc import server
import flwr as fl
import tensorflow as tf
from flwr.server.strategy import FedAvg
import argparse
import socket
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def start_server(num_rounds: int, num_clients: int, fraction_fit: float, server_address=None, port=8080):
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = model.get_weights()
    initial_params = fl.common.weights_to_parameters(weights)
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit, initial_parameters=initial_params)
    if server_address is None:
        # Exposes the server by default on port 8080
        fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})
    else:
        fl.server.start_server(
            server_address=f'{server_address}:{port}',
            strategy=strategy, 
            config={"num_rounds": num_rounds}
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', type=int, required=True)
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--fraction_fit', type=float, required=True)
    parser.add_argument('--port', type=str, required=False, default=8080)

    ip_address = socket.gethostbyname(socket.gethostname())

    print(f'SERVER IP ADDRESS: {ip_address}')

    args = parser.parse_args()

    print(args)

    start_server(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        fraction_fit=args.fraction_fit,
        server_address=ip_address,
        port=args.port
    )
