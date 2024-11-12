import numpy as np
import torch
import torch.nn as nn
import tqdm

import matplotlib.pyplot as plt
from  matplotlib.colors import TwoSlopeNorm
import torch.nn.functional as F
from scipy.stats import bernoulli

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PruneNet(nn.Module):
    '''
    32x32x3 -- CNN --> 32x32x64 -- Max pool --> 10x10x256 --> 2304 -> 384 -> 192 -> 10

    Copied the text from Stothers 2019 into chatgpt:
    
    >I want to implement a convolutional neural network with the following parameters in pytorch: "s 64 5x5 convolutions,
    3x3 max pool, batch normalization, 256 5x5 convolutions, 3x3 max pool, batch normalization, flatten, followed by three dense layers with 382, 194, and 10 units with a softmax on the output. All layers contain rectified linear nonlinearities and categorical
    cross-entropy loss is used."
    
    The kernel at each stage is the same dimensions as the kernel window. For example, in conv1, we start with a 3 channel image and slide a window of size 5x5. The kernel to transform this 3Dx5Wx5H to a single value is another 3Dx5Wx5H kernel which weights each entry in this tensor/matrix prior to summing all the values.
    
    To increase the number of channels (ie 3 to 64), you make 64 of these kernels. Ideally during training each kernel will apply different weights to detect different features.
    
    Thus, to prune kernels, we are removing some of the 64 3Dx5Wx5H kernels. In Stothers 2019 he uses the L2 norm as signal for activity of a kernel.
    '''
    def __init__(self, num_training_iter, num_classes=10, gamma=0.1, verbose=False):
        super().__init__()

        # conv layer 1. input 32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.active_conv1_kernels = nn.Parameter(
            torch.ones(64, requires_grad=False).reshape(-1, 1, 1, 1), 
            requires_grad=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0) # --> 10x10

        # conv layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.active_conv2_kernels = nn.Parameter(
            torch.ones(256, requires_grad=False).reshape(-1, 1, 1, 1),
            requires_grad=False
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

        self.dropout1 = nn.Dropout(0.5) # dropout2d?
        self.dropout2 = nn.Dropout(0.5)

        # initalization
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        # prune proportion
        self.gamma = gamma
        
        self.verbose = verbose
        self.print = print if verbose else lambda *x, **y: None

        self.prune_prob = 1.0
        self.prune_prob_history = []
        self.prune_history = []
        self.conv1_kernel_count_history = []
        self.conv2_kernel_count_history = []
        self.num_training_iter = num_training_iter

    def forward(self, x):
        self.print("x = ", x.shape)
        # with torch.no_grad():
        #     self.conv1.weight = nn.Parameter(self.conv1.weight * self.active_conv1_kernels)
        #     self.conv2.weight = nn.Parameter(self.conv2.weight * self.active_conv2_kernels)

        # 32x32x3 --> 32x32x64 --> 10x10x64
        x = F.relu(self.bn1(self.conv1(x)))
        self.print("x = F.relu(self.bn1(self.conv1(x)))=", x.shape)
        x = self.pool1(x)
        self.print("x = self.pool1(x)", x.shape)

        # 10x10x64 --> 10x10x256 --> 3x3x256
        x = F.relu(self.bn2(self.conv2(x)))
        self.print("x = F.relu(self.bn2(self.conv2(x)))=", x.shape)
        x = self.pool2(x)
        self.print("x = self.pool2(x)", x.shape)

        # 3x3x256 --> 1x2304
        x = torch.flatten(x, start_dim=1)
        self.print("x = torch.flatten(x, start_dim=1)=", x.shape)

        x = F.relu(self.fc1(self.dropout1(x)))
        self.print("x = F.relu(self.fc1(x))=", x.shape)
        x = F.relu(self.fc2(self.dropout2(x)))
        self.print("x = F.relu(self.fc2(x))=", x.shape)
        x = self.fc3(x) # softmax applied during loss computation
        self.print("x = self.fc3(x)=", x.shape)

        if self.verbose:
            assert False
        return x

    def batch_kernel_L2(self, X):
        '''
        X: num kernels x in_channels x kernel width x kernel height (4 dims)
        '''
        with torch.no_grad():
            return X.pow(2).sum(dim=(1, 2, 3)).pow(0.5)
        
    def prune_kernels_low_activity(self):

        conv1_L2 = self.batch_kernel_L2(self.conv1.weight.detach())
        conv2_L2 = self.batch_kernel_L2(self.conv2.weight.detach())

        conv1_alive_idx = torch.argwhere(self.active_conv1_kernels.view(-1) == 1.0).view(-1)
        num_conv1_prune = int(len(conv1_alive_idx) * self.gamma)
        dead_conv1_kernels = np.argsort(conv1_L2[conv1_alive_idx])[:num_conv1_prune]
        self.active_conv1_kernels.view(-1).view(-1)[conv1_alive_idx[dead_conv1_kernels]] = 0.0

        conv2_alive_idx = torch.argwhere(self.active_conv2_kernels.view(-1) == 1.0).view(-1)
        num_conv2_prune = int(len(conv2_alive_idx) * self.gamma)
        dead_conv2_kernels = np.argsort(conv2_L2[conv2_alive_idx])[:num_conv2_prune]
        self.active_conv2_kernels.view(-1).view(-1)[conv2_alive_idx[dead_conv2_kernels]] = 0.0

    def prune_kernels_random(self):

        conv1_alive_idx = torch.argwhere(self.active_conv1_kernels.view(-1) == 1.0).view(-1)
        num_conv1_prune = int(len(conv1_alive_idx) * self.gamma)
        dead_conv1_kernels = conv1_alive_idx[
            np.random.choice(np.arange(len(conv1_alive_idx)), replace=False)
        ]
        self.active_conv1_kernels.view(-1).view(-1)[dead_conv1_kernels] = 0.0

        conv2_alive_idx = torch.argwhere(self.active_conv2_kernels.view(-1) == 1.0).view(-1)
        num_conv2_prune = int(len(conv2_alive_idx) * self.gamma)
        dead_conv2_kernels = conv2_alive_idx[
            np.random.choice(np.arange(len(conv2_alive_idx)), replace=False)
        ]
        self.active_conv2_kernels.view(-1).view(-1)[dead_conv2_kernels] = 0.0

    def update_params(self, method='low_activity'):
        decision = bernoulli.rvs(self.prune_prob, size=1)[0]
        self.prune_prob_history.append(self.prune_prob)

        if method == 'low_activity':
            prune_fn = self.prune_kernels_low_activity
        elif method == 'random':
            prune_fn = self.prune_kernels_random

        if decision == 0:
            prune_fn()

        # update growth parameters
        self.prune_prob = np.max(self.prune_prob - (1.0 / self.num_training_iter), 0)
        self.prune_history.append(decision)
        self.conv1_kernel_count_history.append((self.active_conv1_kernels.view(-1) == 1.0).sum())
        self.conv2_kernel_count_history.append((self.active_conv2_kernels.view(-1) == 1.0).sum())

        return decision
        
        