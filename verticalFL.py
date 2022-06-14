import sys
sys.path.append('../')

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data._utils.collate import default_collate

from typing import List, Tuple
import random
from uuid import uuid4, UUID
import numpy as np
import pandas as pd
import copy
import timeit

import syft as sy

from dataloader import VerticalDataLoader
from src.psi.util import Client, Server
# from util import Client, Server

hook = sy.TorchHook(torch)

class Parser:
    def __init__(self):
        self.epochs = 10
        self.lr = 0.01
        self.seed = 0
        self.input_size = 30 # 30 dimensions
        self.hidden_sizes = [64, 16, 4] # can be altered
        self.output_size = 2 # 0 or 1
        self.batch_size = 128
    
args = Parser()
torch.manual_seed(args.seed)


class VerticalDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, data, targets, *args, **kwargs):
        'Initialization'
        super().__init__(*args, **kwargs)
        self.ids = ids
        self.data = data
        self.targets = targets

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        uuid = self.ids[index]
        if self.data is not None:
            X = self.data[index]
        else:
            X = None
        if self.targets is not None:
            y = self.targets[index]
        else:
            y = None
        return (*filter(lambda x: x is not None, (uuid, X, y)),)
    
    def get_ids(self) -> List[str]:
        """Return a list of the ids of this dataset."""
        return [str(_) for _ in self.ids]
    
    def sort_by_ids(self):
        """Sort the dataset by IDs in ascending order"""
        ids = self.get_ids()
        sorted_idxs = np.argsort(ids)

        if self.data is not None:
            self.data = self.data[sorted_idxs] 

        if self.targets is not None:
            self.targets = self.targets[sorted_idxs]

        self.ids = self.ids[sorted_idxs]

# ----- Load Data ------ #
# Dataset
ids = np.array([uuid4() for i in range(10 ** 4)])
features = torch.randn((10 ** 4, 30))
# features = torch.from_numpy(np.random.randn(10, 30).astype(np.float32))
labels = torch.randint(0, 2, (10 ** 4,))

# Generator
data = VerticalDataset(ids, features, labels)
# data = add_ids(Dataset(features, labels))
dataloader = VerticalDataLoader(data, batch_size=args.batch_size)

# ----- Implement PSI and order the datasets accordingly ----- #
if dataloader.dataloader1.dataset.ids[:].all() != dataloader.dataloader2.dataset.ids[:].all():
    print("Partitioned data is disordered")
    
# Compute private set intersection
client_items = dataloader.dataloader1.dataset.get_ids()
server_items = dataloader.dataloader2.dataset.get_ids()

client = Client(client_items)
server = Server(server_items)

setup, response = server.process_request(client.request, len(client_items))
intersection = client.compute_intersection(setup, response)

# Order data
dataloader.drop_non_intersecting(intersection)
dataloader.sort_by_ids()

if dataloader.dataloader1.dataset.ids[:].all() == dataloader.dataloader2.dataset.ids[:].all():
    print("Partitioned data is aligned")

class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.data = []
        self.remote_tensors = []

    def forward(self, x):
        data = []
        remote_tensors = []

        data.append(self.models[0](x))

        if data[-1].location == self.models[1].location:
            remote_tensors.append(data[-1].detach().requires_grad_())
        else:
            remote_tensors.append(data[-1].detach().move(self.models[1].location).requires_grad_())

        i = 1
        while i < (len(models) - 1):
            data.append(self.models[i](remote_tensors[-1]))

            if data[-1].location == self.models[i + 1].location:
                remote_tensors.append(data[-1].detach().requires_grad_())
            else:
                remote_tensors.append(
                    data[-1].detach().move(self.models[i + 1].location).requires_grad_()
                )

            i += 1

        data.append(self.models[i](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]
    
    def backward(self):
        for i in range(len(models) - 2, -1, -1):
            if self.remote_tensors[i].location == self.data[i].location:
                grads = self.remote_tensors[i].grad.copy()
            else:
                grads = self.remote_tensors[i].grad.copy().move(self.data[i].location)
    
            self.data[i].backward(grads)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


# create workers
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
workers = [alice, bob]

# create models
models = [
    nn.Sequential(
        nn.Linear(args.input_size, args.hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(args.hidden_sizes[0], args.hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(args.hidden_sizes[1], args.hidden_sizes[2]),
        nn.ReLU(),
    ),
    nn.Sequential(nn.Linear(args.hidden_sizes[2], args.output_size), nn.LogSoftmax(dim=1)),
]

# init optimizers
optimizers = [optim.SGD(model.parameters(), lr=args.lr,) for model in models]

# send models to each working node
for model, worker in zip(models, workers):
    model.send(worker)
    
# init splitNN
splitNN = SplitNN(models, optimizers)


def train(features, labels, network):
    
    #1) Zero our grads
    network.zero_grads()
    
    #2) Make a prediction
    pred = network.forward(features)
    
    #3) Figure out how much we missed by
    criterion = nn.MSELoss()
    loss = criterion(pred, labels.float())
    
    #4) Backprop the loss on the end layer
    loss.backward()
    
    #5) Feed Gradients backward through the network
    network.backward()
    
    #6) Change the weights
    network.step()
    
    return loss, pred

start = timeit.default_timer()

for epoch in range(args.epochs):
    running_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for (ids1, features), (ids2, labels) in dataloader:
        # format data
        features = features.send(models[0].location)
        labels = labels.send(models[-1].location)
        labels = labels.view(-1, 1)
        
        # training
        loss, preds = train(features, labels, splitNN)

        # Collect statistics
        running_loss += loss.get()
        correct_preds += preds.max(1)[1].eq(labels).sum().get().item()
        total_preds += preds.get().size(0)

    print(f"Epoch {epoch} - Training loss: {running_loss/len(dataloader):.3f} - Accuracy: {100*correct_preds/total_preds:.3f}")

    stop = timeit.default_timer()
print('Time: ', stop - start)