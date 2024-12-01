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
            # self.trace.append(x)
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


class DrLIMPruneGrowNetwork(PruneGrowNetwork):
    
    def __init__(
        self, 
        gamma,
        init_density,
        num_training_iter,
        low_mapping_dim,
        verbose=False
    ):
        super().__init__(
            gamma,
            init_density,
            num_training_iter,
            verbose=False
        )

        # overwrite architecture
        l1 = BaseDenseLayer(28*28, 16, init_density)
        l2 = BaseDenseLayer(16, low_mapping_dim, init_density)
        self.layers = nn.ModuleList([l1, l2])
        self.total_size = (28*28 * 16) + (16 * 2)

    def forward(self, x):
        """
        Siamese network
        x: a tuple pair of data points
        """
        ds = []
        # Pass both data points through the network
        for d in x:
            d = self.flatten(d)
            for layer in self.layers:
                d = layer(d)
                # self.trace.append(x)
            ds.append(d)
            
        return ds

# Copied from https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/
class ContrastiveLoss(nn.Module):
    def __init__(self, m=2.0, reduction='mean'):
        """
        reduction: mean or sum
        """
        super().__init__()  
        self.m = m  # margin or radius
        self.reduction = reduction
    
    def forward(self, y1, y2, d=0):
        """
        y1: embedding of first data point, batch x final embedding dimension
        y2: embedding of second data point, batch x final embedding dimension
        d = 0 means y1 and y2 are supposed to be same
        d = 1 means y1 and y2 are supposed to be different
        """
        euc_dist = (y1 - y2).pow(2).sum(1).sqrt() # sum along axis 1 (across rows)
        delta = self.m - euc_dist  # distance from the margin
        delta = torch.clamp(delta, min=0.0, max=None) # 0 if >= the margin. positive if < margin
        L = (1 - d) * 0.5 * torch.pow(euc_dist, 2) + d * 0.5 * torch.pow(delta, 2)

        if self.reduction == 'sum':
            return torch.sum(L)
        else:
            return torch.mean(L)
        

def constrative_test_loop(dataloader, model, loss_fn, verbose_print, margin):
    # set to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad(): # no gradients computed
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(*pred, y).item()

            # distance is within margin
            y1, y2 = pred[0], pred[1]
            euc_dist = (y1 - y2).pow(2).sum(1).sqrt()
            
            # similar and within margin, disimilar and larger than margin
            num_correct = (
                ((euc_dist < margin) * (y == 0.0)) + ((euc_dist >= margin) * (y == 1.0)) # 1 for true, 0 for false
            ).int()
            correct += torch.sum(num_correct).item()

    print(correct)
    test_loss /= num_batches
    correct /= size
    
    verbose_print(f"Test Error \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>3f}\n")
    return correct, test_loss
    
        