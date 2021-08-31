import os
import time
from multiprocessing import Process
from typing import Tuple
from collections import OrderedDict

import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg
import torch
import dataset
import json
from torchrppg.models import get_model
from torchrppg.optim import get_optimizer
from torchrppg.loss import loss_fn

DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

with open('params.json') as f:
    jsonObject = json.load(f)
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
    available_gpu_number = params.get("available_gpu")


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)

    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})


def start_client(dataset: DATASET, client_number: int, total_gpu: int) -> None:
    """Start a single client with the provided dataset."""
    total_gpu = len(available_gpu_number) if len(available_gpu_number) < total_gpu else total_gpu

    DEVICE = torch.device(
        ("cuda:" + available_gpu_number[(client_number + 1) % total_gpu]) if torch.cuda.is_available() else "cpu")

    net = get_model(model_params["name"]).to(DEVICE)

    trainloader, testloader = dataset

    class RPPGClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            for k in state_dict.keys():
                v = state_dict[k]
                if k.__contains__("num_batches_tracked") and len(v.shape) == 1 and v.shape[0] == 0:
                    state_dict[k] = torch.tensor([0], dtype=torch.long)
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, hyper_params["epochs"], DEVICE)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=RPPGClient())


def train(net, trainloader, epochs, device: torch.device):
    """Train the network on the training set."""
    optimizer = get_optimizer(net.parameters(), hyper_params["learning_rate"], hyper_params["optimizer"])

    criterion = loss_fn(hyper_params["loss_fn"])
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            labels = labels.view(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = loss_fn(hyper_params["loss_fn"])
    loss = 0.0
    net.eval()
    device = next(net.parameters()).get_device()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            labels = labels.view(outputs.shape)
            error = criterion(outputs, labels).item()
            loss += error
            _, predicted = torch.max(outputs.data, 1)
    return loss, 0


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""
    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients,
                              batch_size=params['train_batch_size'],
                              shuffle=params['train_shuffle'],
                              model_name=model_params['name'],
                              dataset_name=params['dataset_name'])

    # Start all the clients
    for client_number, partition in enumerate(partitions):
        client_process = Process(target=start_client,
                                 args=(partition, client_number, params['total_gpu'],))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_simulation(num_rounds=params['num_rounds'],
                   num_clients=params['num_clients'],
                   fraction_fit=params['fraction_fit'])
