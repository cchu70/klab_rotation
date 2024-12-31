import numpy as np
import torch
import torch.nn as nn
import tqdm

import matplotlib.pyplot as plt
from  matplotlib.colors import TwoSlopeNorm
import torch.nn.functional as F
from scipy.stats import bernoulli

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv2dWithActivity(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    ):
        super(Conv2dWithActivity, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        # Initialize the activity array
        activity = torch.ones(out_channels, dtype=torch.float32)
        self.activity = nn.Parameter(activity, requires_grad=False)

    def forward(self, input):
        # Apply the activity array to the kernel
        kernel = self.weight.to(input.device) * self.activity.to(input.device).view(self.out_channels, 1, 1, 1)
        return self._conv_forward(input, kernel.to(input.device), self.bias.to(input.device)).to(input.device)
        
class MiniAlexNet(nn.Module):
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
    def __init__(self, num_training_iter, num_classes=10, gamma=0.1, verbose=False, random_seed=1, in_channels=3, device='cuda:0'):
        super().__init__()

        # conv layer 1. input 32x32
        self.conv1 = Conv2dWithActivity(
            in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2
        ) #.to(device)
        self.bn1 = nn.BatchNorm2d(64) #.to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0) #.to(device) # --> 10x10

        # conv layer 2
        self.conv2 = Conv2dWithActivity(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=1) #.to(device)
        self.bn2 = nn.BatchNorm2d(256) #.to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) #.to(device)

        # fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 384) #.to(device)
        self.fc2 = nn.Linear(384, 192) #.to(device)
        self.fc3 = nn.Linear(192, num_classes) #.to(device)

        self.dropout1 = nn.Dropout(0.5) #.to(device) # dropout2d?
        self.dropout2 = nn.Dropout(0.5) #.to(device)

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

        self.prune_prob = 0.0
        self.prune_prob_history = []
        self.prune_history = []
        self.conv1_kernel_count_history = []
        self.conv2_kernel_count_history = []
        self.num_training_iter = num_training_iter

        self.random_seed = random_seed
        np.random.seed(seed=random_seed)

        self.use_grow_prune_prob = False

    def get_attribute_dict(self):
        attr_dict = {
            'gamma': self.gamma,
            'prune_prob': self.prune_prob,
            'prune_history': self.prune_history,
            'conv1_kernel_count_history': self.conv1_kernel_count_history,
            'conv2_kernel_count_history': self.conv2_kernel_count_history,
            'total_conv_kernel_count_history': list(np.array(self.conv1_kernel_count_history) + np.array(self.conv2_kernel_count_history)),
            'prune_prob_history': self.prune_prob_history,
            'num_training_iter': self.num_training_iter,
            'random_seed': self.random_seed
        }
        return attr_dict

    def forward(self, x):
        self.print("x = ", x.shape)

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
        x = self.fc3(x) # softmax applied during loss computation if doing classification (ie 10 classes for CIFAR10)
        self.print("x = self.fc3(x)=", x.shape)

        if self.verbose:
            assert False
        return x

    def prune_activity_conv2d(self, conv2d):
        pass

    def prune_kernels(self):
        print("no pruning")
        pass

    def update_params(self):
        decision = bernoulli.rvs(self.prune_prob, size=1)[0]
        self.prune_prob_history.append(self.prune_prob)
        
        if self.use_grow_prune_prob:
            if decision == 1.0:
                self.prune_kernels()
            self.prune_prob = np.min([self.prune_prob + (1.0 / self.num_training_iter), 1.0])

        # update growth parameters
        self.prune_history.append(decision)
        self.conv1_kernel_count_history.append((self.conv1.activity.view(-1) == 1.0).sum())
        self.conv2_kernel_count_history.append((self.conv2.activity.view(-1) == 1.0).sum())

        return decision

class RandomPruneNet(MiniAlexNet):

    def __init__(self, num_training_iter, num_classes=10, gamma=0.1, verbose=False, random_seed=1, in_channels=3, device="cuda:0"):
        super().__init__(
            num_training_iter=num_training_iter, num_classes=num_classes, gamma=gamma, 
            verbose=verbose, random_seed=random_seed, in_channels=in_channels, device=device
        )
        self.use_grow_prune_prob = True

    def prune_activity_conv2d(self, conv2d):
        alive_idx = torch.argwhere(conv2d.activity == 1.0)
        num_prune = int(len(alive_idx) * self.gamma)
        dead_idx = alive_idx[
            np.random.choice(np.arange(len(alive_idx)), size=num_prune, replace=False)
        ]
        conv2d.activity[dead_idx] = 0.0
        

    def prune_kernels(self):
        self.print("random pruning")

        for conv2d in [self.conv1, self.conv2]:
            self.prune_activity_conv2d(conv2d)


class ActivityPruneNet(MiniAlexNet):

    def __init__(self, num_training_iter, num_classes=10, gamma=0.1, verbose=False, random_seed=1, in_channels=3, device="cuda:0"):
        super().__init__(
            num_training_iter=num_training_iter, num_classes=num_classes, gamma=gamma, 
            verbose=verbose, random_seed=random_seed, in_channels=in_channels, device=device
        )
        self.use_grow_prune_prob = True
    
    def batch_kernel_L2(self, X):
        '''
        X: num kernels x in_channels x kernel width x kernel height (4 dims)
        '''
        with torch.no_grad():
            return X.pow(2).sum(dim=(1, 2, 3)).pow(0.5)

    def prune_activity_conv2d(self, conv2d):
        with torch.no_grad():
            L2 = self.batch_kernel_L2(conv2d.weight.detach())
            self.print(L2.shape)
            
            alive_idx = torch.argwhere(conv2d.activity == 1.0)
            num_prune = int(len(alive_idx) * self.gamma)
    
            dead_idx = np.argsort(L2[alive_idx.view(-1)])[:num_prune]
            conv2d.activity[alive_idx.view(-1)[dead_idx]] = 0.0
        
    def prune_kernels(self):
        self.print("activity pruning")

        for conv2d in [self.conv1, self.conv2]:
            self.prune_activity_conv2d(conv2d)
        
class DrLIMMiniAlexNet(MiniAlexNet):
    """
    input x is a tuple containing pairs of images/tensors
    """
    def forward(self, x):
        """
        Siamese network
        x: a tuple pair of data points
        """
        ds = []
        # Pass both data points through the network
        for d in x:
            output = super(DrLIMMiniAlexNet, self).forward(d)
            ds.append(output)
        return ds

class DrLIMActivityPruneNet(RandomPruneNet):

    """
    input x is a tuple containing pairs of images/tensors
    """
    def forward(self, x):
        """
        Siamese network
        x: a tuple pair of data points
        """
        ds = []
        # Pass both data points through the network
        for d in x:
            output = super(DrLIMActivityPruneNet, self).forward(d)
            ds.append(output)
        return ds

class DrLIMRandomPruneNet(ActivityPruneNet):
    """
    input x is a tuple containing pairs of images/tensors
    """
    def forward(self, x):
        """
        Siamese network
        x: a tuple pair of data points
        """
        ds = []
        # Pass both data points through the network
        for d in x:
            output = super(DrLIMRandomPruneNet, self).forward(d)
            ds.append(output)
        return ds  
        