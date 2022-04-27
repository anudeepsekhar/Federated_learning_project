from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import FedAvg,Strategy
from flwr.server.server import Server
from ray_client_proxy import RayClientProxy
from flwr.common.logger import log
from logging import INFO
import ray
import sys
def start_simulation(client_fn = None, num_clients = None, clients_ids = None, client_resources = None, num_rounds = None, strategy = None):
    """
    A function to start the simulation.
    """
    keep_initialized = False
    if clients_ids is not None:
        if (num_clients is not None) and (len(clients_ids) != num_clients):
            sys.exit("Issue with the Client Ids")
        else:
            cids = clients_ids
    else:
        if num_clients is None:
            sys.exit("Number of Clients is not being passed you fool!")
        else:
            cids = [str(x) for x in range(num_clients)]

    #Default arguments for Ray initialization
    ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    if ray.is_initialized() and not keep_initialized:
        ray.shutdown()

    ray.init(**ray_init_args)
    client_manager = SimpleClientManager()
    if strategy is None:
        strategy = FedAvg()
    server = Server(client_manager = client_manager, strategy = strategy)
    resources = client_resources if client_resources is not None else {}

    for cid in cids:
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=cid,
            resources=resources,
        )
        server.client_manager().register(client = client_proxy)


    #Training the model
    hist = server.fit(num_rounds = num_rounds)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))
    server.disconnect_all_clients()
    return hist