import numpy as np
import pandas as pd
import tqdm
import torch
from torch import nn
from scipy.stats import bernoulli

class BaseDenseLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_density: int,
    ):
        """
        W: Weight matrix
        A: Alive matrix
        b: Bias vector
        act: activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        with torch.no_grad():
            num_init = int(init_density * input_dim * output_dim)
            A = torch.zeros((input_dim, output_dim), dtype=torch.float, requires_grad=False)
            rand_sparse_idx = np.random.choice(range(int(A.view(-1).shape[0])), size=num_init)
            A.view(-1)[rand_sparse_idx] = 1.0
    
            W = nn.init.kaiming_uniform_(torch.empty((input_dim, output_dim), dtype=torch.float))
            W.view(-1)[(A != 1.0).view(-1)] = 0.0

        self.A = nn.Parameter(A, requires_grad=False)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(output_dim, dtype=torch.float))
        self.act = nn.ReLU()
        
    def forward(self, x):
        """
        W is masked by A to enforce growth/pruning
        """
        masked_W = self.W * self.A
        return self.act(x @ masked_W + self.b)

class PredictionHead(nn.Module):
    """
    W: Weight matrix
    A: Alive matrix
    b: Bias vector

    No Activation
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        init_density: int,
    ):
        super().__init__()
        with torch.no_grad():
            num_init = int(init_density * input_dim * num_classes)
            A = torch.zeros((input_dim, num_classes), dtype=torch.float, requires_grad=False)
            rand_sparse_idx = np.random.choice(range(int(A.view(-1).shape[0])), size=num_init)
            A.view(-1)[rand_sparse_idx] = 1.0
    
            W = nn.init.kaiming_uniform_(torch.empty((input_dim, num_classes), dtype=torch.float))
            W.view(-1)[(A != 1.0).view(-1)] = 0.0

        self.A = nn.Parameter(A, requires_grad=False)
        self.W = nn.Parameter(W, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_classes, dtype=torch.float))
        
    def forward(self, x):
        """
        W is masked by A to enforce growth/pruning
        """
        masked_W = self.W * self.A
        return x @ masked_W + self.b
        

class PruneGrowNetwork(nn.Module):
    def __init__(
        self,
        gamma,
        init_density,
        num_training_iter,
        verbose=False
    ):
        super().__init__()
        l1 = BaseDenseLayer(28*28, 16, init_density)
        l2 = PredictionHead(16, 10, init_density)
        self.layers = nn.ModuleList([l1, l2])
        self.total_size = (28*28 * 16) + (16 * 10)

        self.flatten = nn.Flatten()
        
        self.trace = []
        self.gamma = gamma
        self.grow_prob = 1.0 # start with growth

        # TODO: self.register_buffer()
        self.grow_prune_history = []
        self.synapse_count_history = []
        self.grow_prob_history = []
        self.num_training_iter = num_training_iter
        
    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            self.trace.append(x)
        return x

    def synapse_size(self):
        size = 0
        for layer in self.layers:
            size += layer.A.sum().item()
        return size

    def update_params(self):
        decision = bernoulli.rvs(self.grow_prob, size=1)[0]
        self.grow_prob_history.append(self.grow_prob)

        if decision == 1:
            self.rand_grow()
        elif decision == 0:
            self.activity_prune()

        # update growth parameters
        self.grow_prob = np.max(self.grow_prob - (1.0 / self.num_training_iter), 0)
        self.grow_prune_history.append(decision)
        self.synapse_count_history.append(self.synapse_size())
        return decision

    def rand_grow(self):
        
        for layer in self.layers:
            dead_idx = torch.argwhere(layer.A == 0)

            num_resurrect = int(np.ceil(len(dead_idx) * self.gamma))
            rand_dead_idx = dead_idx[np.random.choice(range(len(dead_idx)), size=num_resurrect), :]

            for idx in rand_dead_idx:
                layer.A[*idx] = 1

    def activity_prune(self):
        with torch.no_grad():
            for layer in self.layers:
                
                alive_idx = torch.argwhere(layer.A.view(-1) == 1).view(-1)
    
                num_prune = int(np.ceil(len(alive_idx) * self.gamma))
                prune_alive_idx = torch.argsort(torch.abs(layer.W.view(-1)[alive_idx]))[:num_prune]

                layer.A.view(-1)[alive_idx[prune_alive_idx]] = 0.0
                layer.W.view(-1)[alive_idx[prune_alive_idx]] = 0.0

        
        

           
    
        