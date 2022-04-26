import flwr as fl
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_dataset
from models import CNNMnist, CNNCifar
import numpy as np
from client import FlowerClient
import argparse
import subprocess, sys

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10,
                        help = "number of rounds of training")
    parser.add_argument('--local_epoch', type = int, default = 1,
                        help = "number of local epochs for training")
    parser.add_argument('--local_bs', type = int, default = 64,\
                        help = "batch size")
    parser.add_argument('--dataset', type = str, default = 'cifar',\
                        help = "The dataset to train and evaluate")
    parser.add_argument('--frac_fit', type = float, default = 1.0,\
                        help = 'The fraction of clients to choose for training')
    parser.add_argument('--frac_eval', type = float, default = 0.5,\
                        help = 'The fraction of clients for evaluation')
    parser.add_argument('--min_fit_clients', type = int, default = 10,\
                        help = 'The minimum number of clients for training')
    parser.add_argument('--min_eval_clients', type = int, default = 10,\
                        help = 'The minimum number of clients for evaluation')
    parser.add_argument('--num_users', type = int, default = 100,\
                        help = 'The number of users')
    parser.add_argument('--min_available_clients', type = int, default = 10,\
                        help = 'Wait for these many clients to be available')
    parser.add_argument('--iid', type = int, default = 0, \
                        help = '0 for iid and 1 for non iid')
    args = parser.parse_args()
    #Get the train_dataset, test_dataset, user_group ids for training and validation
    train_dataset, test_dataset, user_groups_train, user_groups_val = get_dataset(args)
    trainloaders = []
    valloaders = []

    #A loop to put all the trainloaders in a list and valloaders in another list
    for user in range(args.num_users):
        trainloaders.append(DataLoader(DatasetSplit(train_dataset, user_groups_train[user]),\
            batch_size = args.local_bs, shuffle = True, num_workers = 2))

        valloaders.append(DataLoader(DatasetSplit(test_dataset, user_groups_val[user]),\
            batch_size = args.local_bs, shuffle = False, num_workers = 2))

    #Setting the dataset
    if args.dataset == 'mnist':
        global_model = CNNMnist(args).to(DEVICE)
    elif args.dataset == 'cifar':
        global_model = CNNCifar(args).to(DEVICE)

    #A function to manage the clients.
    def client_fn(cid):
        """
        A custom function to manage all the clients.
        :param cid: The id for each client
        :return FlowerClient: Child of the FlowerClient class
        """
        net = global_model
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader)

    def fit_config(rnd):
        """
        Return training configuration dict for each round.
        custom configuration for local training.
        """
        config = {
            "local_epochs" : args.local_epoch,
            "current_rnd" : rnd
        }
        return config

    #Setting the strategy for the Clients
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = args.frac_fit,
        fraction_eval = args.frac_eval,
        min_fit_clients = args.min_fit_clients,
        min_eval_clients = args.min_eval_clients,
        min_available_clients = args.min_available_clients,
        on_fit_config_fn = fit_config
    )

    #Start the Simulation
    fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = args.num_users,
        num_rounds = args.epochs,
        strategy = strategy
    )
