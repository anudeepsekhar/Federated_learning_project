from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
logger = SummaryWriter('./logs')
import csv
from file import return_writer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(net, cid, curr_rnd, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    epoch_loss_list = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        epoch_loss_list.append(epoch_loss)
        data  = [f'{curr_rnd}', f'{epoch}', f'{cid}', f'{epoch_loss}', f'{epoch_acc}']
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return sum(epoch_loss_list)/len(epoch_loss_list)

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def get_parameters(net) -> List[np.ndarray]:
    """
    A function which the server uses to get the paramaters from the clients.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """
    A function which sets the parameters of the clients from the servers.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    """
    A class for the Flower Client.
    """
    def __init__(self, cid, model, trainloader, valloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader


    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        try:
            curr_round = config["current_rnd"]
            set_parameters(self.model, parameters)
            train_loss = train(self.model, self.cid, curr_round, self.trainloader, epochs = config["local_epochs"])
        except Exception as e:
            print("FAILURE", e)
        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        try:
            loss, accuracy = test(self.model, self.valloader)
            logger.add_scalars(f'cid : {self.cid}', {'loss_eval' : loss, 'accuracy_eval' : accuracy},0)
        except Exception as e:
            print("FAILURE", e)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss" : float(loss)}
