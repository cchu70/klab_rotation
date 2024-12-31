from .simple_pruning_growth_model import PruneGrowNetwork, DrLIMPruneGrowNetwork, ContrastiveLoss, constrative_test_loop
from .CNN_pruning import MiniAlexNet, ActivityPruneNet, RandomPruneNet, DrLIMMiniAlexNet, DrLIMActivityPruneNet, DrLIMRandomPruneNet
from .training_testing_loop import full_train, save_model_attr, format_training_outputs
from .load_MNIST import get_mnist_pairs_loader
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import date
import pickle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import os
import gzip
import warnings
from sklearn import metrics

# See simple_mlp_unsupervised_train
class TrainingResults:

    def __init__(self, output_dir, desc_split_idx=-2, param_split_idx=-1, param_dict=None, desc=None):
        self.output_dir = output_dir
        if desc is not None:
            self.desc = desc
        else:
            self.desc = output_dir.split('/')[desc_split_idx]

        if param_dict is not None:
            self.params = param_dict
        else:            
            param_str = output_dir.split('/')[param_split_idx]
            self.params = {s.split("-")[0]: s.split("-")[1] for s in param_str.split("_")}
        self.stack_training_losses_df = pd.read_csv(f"{self.output_dir}/stack_training_losses.tsv", sep='\t')
        self.stack_val_losses_df = pd.read_csv(f"{self.output_dir}/stack_val_losses.tsv", sep='\t')
        self.test_df = pd.read_csv(f"{self.output_dir}/test_err_loss.tsv", sep='\t')

        with open(f"{self.output_dir}/model_attr.pkl", "rb") as fh:
            self.model_attr = pickle.load(fh)

        model_state_dicts_pkl = f"{self.output_dir}/model_state_dicts.pkl"
        model_state_dicts_gz = f"{self.output_dir}/model_state_dicts.pkl.gz"
        model_state_dicts_txt = f"{self.output_dir}/model_state_dicts.txt"
        if os.path.exists(model_state_dicts_pkl):
            with open(model_state_dicts_pkl, "rb") as fh:
                self.model_state_dicts = pickle.load(fh)
    
        elif os.path.exists(model_state_dicts_gz):
            with gzip.open(model_state_dicts_gz, 'rb') as fh:
                self.model_state_dicts = pickle.load(fh)

        elif os.path.exists(model_state_dicts_txt):
            with open(model_state_dicts_txt, 'rt') as fh:
                epoch_model_state_dict_fns = [line.rstrip() for line in fh]

            self.model_state_dicts = {}
            for fn in epoch_model_state_dict_fns:
                epoch = int(fn.split("/")[-1].split(".")[0])
                with gzip.open(fn, 'rb') as fh:
                    self.model_state_dicts[epoch] = pickle.load(fh)

        else:
            warnings.warn(f"Directory does not contain model state file. Not loading model_state_dicts")
        
        self.prune_history_attr_name = 'grow_prune_history'
        self.prune_ylabel = 'Grow/prune'
        self.model_size_history_attr_name = 'synapse_count_history'

    def plot_training_losses(self, alpha=1.0):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(self.stack_training_losses_df.reset_index(), x='index', y='loss', label='Training loss', alpha=alpha, c='gray', s=4, ax=ax)
        sns.lineplot(self.stack_val_losses_df.reset_index(), x='index', y='loss',  label='Validation loss', c='red', ax=ax)
        plt.xticks([])
        plt.xlabel('Training and validation epoch/batch')
        plt.yscale('log')
        plt.title(f"{self.desc} Training Losses")
        return fig, ax
    
    def plot_pruning(self, figsize=(10, 5), height_ratios=[1, 5, 5], norm_size=None, test_err_col='test_accuracy'):
        """
        test_err_col: 'test_err' (legacy) or 'test_accuracy'. See model.test_df for column names
        """

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, height_ratios=height_ratios)
        axes[0].imshow(np.array(np.array(self.model_attr[self.prune_history_attr_name]).reshape(1, -1)), cmap='gray', aspect='auto')
        axes[0].set_yticks([])
        axes[0].set_ylabel(self.prune_ylabel, rotation=0, ha='right')

        if norm_size:
            axes[1].plot([x/norm_size for x in self.model_attr[self.model_size_history_attr_name][1:]], c='k') 
            axes[1].set_ylabel("Total model size")
            axes[1].set_ylim(0, 1)
        else:
            axes[1].plot(self.model_attr[self.model_size_history_attr_name][1:], c='k') 
            axes[1].set_ylabel("Total model size")

        sns.scatterplot(self.test_df.reset_index(), x='epoch', y=test_err_col, ax=axes[2], c='k', s=5) 
        axes[2].set_ylim(0, 1.0)
        plt.tight_layout()
        plt.title(f"{self.desc} Pruning History")
        return fig, axes
    
    def plot_model_size_vs_performance(self, test_err_col='test_accuracy', **figkwargs):
        fig, ax = plt.subplots(**figkwargs)
        sns.scatterplot(
            x=self.model_attr[self.model_size_history_attr_name][1:],
            y=self.test_df[test_err_col],
            s=5,
            ax=ax
            # hue=prune_full_results.test_df['test_accuracy'],
            # palette='rocket_r'
        )
        plt.ylabel("Test Accuracy")
        plt.xlabel("Model size")
        return fig, ax

    
    def set_trained_model(self, epoch: int):
        pass
    

class SupervisedTrainingResults(TrainingResults):
    def plot_confusion_matrix(self, dataloader, labels=np.arange(10)):
        with torch.no_grad():
            self.model.eval() # not training
            ys = []
            pred_ys = []
            for X, y in dataloader:
                pred_y = self.model(X)
                ys.append(y)
                pred_ys.append(torch.argmax(pred_y, axis=1))

        ys = torch.cat(ys)
        pred_ys = torch.cat(pred_ys)

        confusion_matrix = metrics.confusion_matrix(y_true=ys, y_pred=pred_ys, labels=labels)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
        disp.plot()
        plt.title(f"Epoch {self.epoch_num}")
        plt.show()

class MLPSupervisedTrainingResults(SupervisedTrainingResults):

    def set_trained_model(self, epoch: int):
        """
        epoch: int
            Epoch in training (see self.model_state_dicts)
        """
            
        self.model = PruneGrowNetwork(
            gamma=0.1, init_density=float(self.params['id']), 
            num_training_iter=float(self.params['nti']), use_grow_prune_prob=bool(self.params['ugpp'])
        )
        self.model.load_state_dict(self.model_state_dicts[epoch])
        self.model.eval()
        self.epoch_num = epoch 

class CNNSupervisedTrainingResults(SupervisedTrainingResults):

    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.prune_history_attr_name = 'prune_history'
        self.prune_ylabel = 'Prune'
        self.model_size_history_attr_name = 'total_conv_kernel_count_history'

    def set_trained_model(self, epoch: int):
        """
        epoch: int
            Epoch in training (see self.model_state_dicts)
        """
        prune_model_type = self.params['pmt']

        model_args = dict(
            num_training_iter=int(self.params['nti']),
            num_classes=int(self.params['lmd']),
            gamma=float(self.params['g']),
            verbose=False, 
            random_seed=int(self.params['s']),
            in_channels=int(self.params['ic']),
        )

        if prune_model_type=="NoPrune":
            model = MiniAlexNet(**model_args)
        elif prune_model_type=="Activity":
            model = ActivityPruneNet(**model_args)
        elif prune_model_type=="Random":
            model = RandomPruneNet(**model_args)
        else:
            raise ValueError(f"prune_model={prune_model_type} is not valid. Options are ['NoPrune', 'Activity', 'Random']")

        self.model = model
        self.model.load_state_dict(self.model_state_dicts[epoch])
        self.model.eval()
        self.epoch_num = epoch 


class UnsupervisedTrainingResults(TrainingResults):
    def plot_pairs(self, pair_dataloader, figsize=(5, 5)):
        self.model.eval()
        colors = {1: 'ro--', 0: 'ko--'}
        fig, ax = plt.subplots(figsize=figsize)
        with torch.no_grad():
            for X, y in pair_dataloader:
                pred = self.model(X)
                for i in range(len(pred[0])):
                    ax.plot([pred[0][i, 0], pred[1][i, 0]], [pred[0][i, 1], pred[1][i, 1]], colors[y[i].item()])

            custom_lines = [Line2D([0], [0], color='r', linestyle='--'),
                            Line2D([0], [0], color='k', linestyle='--')]
            plt.legend(custom_lines, ['Similar', 'Dissimilar'], title='Data pairs')
            
            return fig, ax

    def plot_image_embeddings(self, pair_dataloader, xlim=(-6, 6), ylim=(-4, 4), image_min_value=-0.1, num_pairs_per_batch=5, figsize=(5, 5)):
        """
        xlim, ylim: tuple for limits
        """
        self.model.eval()
        colors = {1: 'red', 0: 'black'}  # Solid colors for points
        line_styles = {1: 'r--', 0: 'k--'}  # Dashed lines for connections
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel('Embedding dimension 1')
        ax.set_ylabel('Embedding dimension 2')
        ax.set_title('2D Embeddings of Image Pairs')
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)

        def plot_img_embedding(img, x, y):
            img[img < image_min_value] = np.nan
            im = OffsetImage(img, cmap='gray_r', zoom=1.0)
            return AnnotationBbox(im, (x, y), xycoords='data', frameon=False)

        with torch.no_grad():
            for X, y in pair_dataloader:
                pred = self.model(X)

                for i in range(len(pred[0])):
                    # Plot points for both images in the pair
                    # plt.scatter(pred[0][i, 0], pred[0][i, 1], c=colors[y[i].item()], alpha=0.6)
                    # plt.scatter(pred[1][i, 0], pred[1][i, 1], c=colors[y[i].item()], alpha=0.6)
                    
                    ab = plot_img_embedding(X[0][i].permute(1, 2, 0).numpy(), pred[0][i, 0].item(), pred[0][i, 1].item())
                    ax.add_artist(ab)

                    ab = plot_img_embedding(X[1][i].permute(1, 2, 0).numpy(), pred[1][i, 0].item(), pred[1][i, 1].item())
                    ax.add_artist(ab)
                    
                    if i > num_pairs_per_batch:
                        break
                        
            return fig, ax

class MLPUnsupervisedTrainingResults(UnsupervisedTrainingResults):

    def set_trained_model(self, epoch: int):
        """
        epoch: int
            Epoch in training (see self.model_state_dicts)
        """
        prediction_act_type = self.params['pat']
        if prediction_act_type == "linear":
            prediction_act = lambda x: x
        if prediction_act_type == "Tanh":
            prediction_act = nn.Tanh()
            
        self.model = DrLIMPruneGrowNetwork(
            gamma=0.1, init_density=float(self.params['id']), num_training_iter=int(self.params['nti']),
            low_mapping_dim=int(self.params['lmd']), prediction_act=prediction_act, use_grow_prune_prob=bool(self.params['ugpp'])
        )
        self.model.load_state_dict(self.model_state_dicts[epoch])
        self.model.eval()
        self.epoch_num = epoch 


class CNNUnsupervisedTrainingResults(UnsupervisedTrainingResults):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.prune_history_attr_name = 'prune_history'
        self.prune_ylabel = 'Prune'
        self.model_size_history_attr_name = 'total_conv_kernel_count_history'

    def set_trained_model(self, epoch: int, device='cpu'):
        """
        epoch: int
            Epoch in training (see self.model_state_dicts)
        """
        model_args = dict(
            num_training_iter=int(self.params['nti']),
            num_classes=int(self.params['lmd']),
            gamma=float(self.params['g']),
            verbose=False, 
            random_seed=int(self.params['s']),
            in_channels=1, # using MNIST dataset
            device=device,
        )
        
        prune_model_type = self.params['pmt']
        if prune_model_type=="NoPrune":
            self.model = DrLIMMiniAlexNet(**model_args)
        elif prune_model_type=="Activity":
            self.model = DrLIMActivityPruneNet(**model_args)
        elif prune_model_type=="Random":
            self.model = DrLIMRandomPruneNet(**model_args)
        else:
            raise ValueError(f"prune_model={prune_model_type} is not valid. Options are []'None', 'Activity', 'Random']")

        self.model.load_state_dict(self.model_state_dicts[epoch])
        self.model.eval()
        self.epoch_num = epoch 