from simple_pruning_growth_model import DrLIMPruneGrowNetwork, ContrastiveLoss, constrative_test_loop
from training_testing_loop import full_train, save_model_attr, format_training_outputs
from load_MNIST import get_mnist_pairs_loader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from datetime import date
import pickle
import os
import argparse
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

class MLPTrainingResults:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        param_str = output_dir.split('/')[-1]
        self.params = {s.split("-")[0]: s.split("-")[1] for s in param_str.split("_")}
        self.stack_training_losses_df = pd.read_csv(f"{self.output_dir}/stack_training_losses.tsv", sep='\t')
        self.stack_val_losses_df = pd.read_csv(f"{self.output_dir}/stack_val_losses.tsv", sep='\t')
        self.test_df = pd.read_csv(f"{self.output_dir}/test_err_loss.tsv", sep='\t')

        with open(f"{self.output_dir}/model_attr.pkl", "rb") as fh:
            self.model_attr = pickle.load(fh)

        with open(f"{self.output_dir}/model_state_dicts.pkl", "rb") as fh:
            self.model_state_dicts = pickle.load(fh)

    def plot_training_losses(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(self.stack_training_losses_df.reset_index(), x='index', y='loss', label='Training loss', alpha=1.0, c='gray', s=4, ax=ax)
        sns.lineplot(self.stack_val_losses_df.reset_index(), x='index', y='loss',  label='Validation loss', c='red', ax=ax)
        plt.xticks([])
        plt.xlabel('Training and validation epoch/batch')
        plt.yscale('log')
        return fig, ax
    
    def plot_pruning(self):
        fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True, height_ratios=[1, 5, 5])
        axes[0].imshow(np.array(np.array(self.model_attr['grow_prune_history']).reshape(1, -1)), cmap='gray')
        axes[0].set_yticks([])
        axes[0].set_ylabel("Grow/prune", rotation=0, ha='right')

        axes[1].plot(self.model_attr['synapse_count_history'], c='k') 
        axes[1].set_ylabel("Total model size")

        sns.scatterplot(self.test_df.reset_index(), x='epoch', y='test_err', ax=axes[2], c='k', s=5) 
        axes[2].set_ylim(0, 1.0)
        plt.tight_layout()
        return fig, axes

    def set_trained_model(self, epoch: int):
        """
        epoch: int
            Epoch in training (see self.model_state_dicts)
        """
        self.DrLIM_model = DrLIMPruneGrowNetwork(
            gamma=0.1, init_density=0.5, num_training_iter=100,
            low_mapping_dim=2, prediction_act=lambda x: x, use_grow_prune_prob=True
        )
        self.DrLIM_model.load_state_dict(self.model_state_dicts[epoch])
        self.epoch_num = epoch

    def plot_pairs(self, pair_dataloader):
        self.DrLIM_model.eval()
        colors = {1: 'ro--', 0: 'ko--'}
        with torch.no_grad():
            for X, y in pair_dataloader:
                pred = self.DrLIM_model(X)
                for i in range(len(pred[0])):
                    plt.plot([pred[0][i, 0], pred[1][i, 0]], [pred[0][i, 1], pred[1][i, 1]], colors[y[i].item()])
            plt.show()

    def plot_image_embeddings(self, pair_dataloader, xlim=(-6, 6), ylim=(-4, 4), image_min_value=-0.1):
        """
        xlim, ylim: tuple for limits
        """
        self.DrLIM_model.eval()
        colors = {1: 'red', 0: 'black'}  # Solid colors for points
        line_styles = {1: 'r--', 0: 'k--'}  # Dashed lines for connections
        fig, ax = plt.subplots()

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
            artists = []
            for X, y in pair_dataloader:
                pred = self.DrLIM_model(X)

                for i in range(len(pred[0])):
                    # Plot points for both images in the pair
                    # plt.scatter(pred[0][i, 0], pred[0][i, 1], c=colors[y[i].item()], alpha=0.6)
                    # plt.scatter(pred[1][i, 0], pred[1][i, 1], c=colors[y[i].item()], alpha=0.6)
                    
                    ab = plot_img_embedding(X[0][i].permute(1, 2, 0).numpy(), pred[0][i, 0].item(), pred[0][i, 1].item())
                    ax.add_artist(ab)

                    ab = plot_img_embedding(X[1][i].permute(1, 2, 0).numpy(), pred[1][i, 0].item(), pred[1][i, 1].item())
                    ax.add_artist(ab)
                    
                break
                        
            return fig, ax

        

def main(
    batch_size=32, subset_fraction=0.5, selected_labels=[4,9], validation_ratio=6,
    init_density=0.5, num_training_iter=100, low_mapping_dim=2, prediction_act_type="linear", margin=5, use_grow_prune_prob=False, 
    learning_rate=1e-3, 
    output_dir=None,
    seed=42,
):
    
    if prediction_act_type == "linear":
        prediction_act = lambda x: x
    if prediction_act_type == "Tanh":
        prediction_act = nn.Tanh()

    if output_dir is None:
        raise ValueError("Set an output directory")
    # get dataloaders
    train_pair_dataloader, val_pair_dataloader = get_mnist_pairs_loader(
        batch_size=batch_size, train=True, subset_fraction=subset_fraction, validation_ratio=validation_ratio, seed=seed, 
        selected_labels=selected_labels
    )
    test_pair_dataloader = get_mnist_pairs_loader(
        batch_size=batch_size, train=False, subset_fraction=subset_fraction, 
        selected_labels=selected_labels
    )

    DrLIM_model = DrLIMPruneGrowNetwork(
        gamma=0.1, init_density=init_density, num_training_iter=num_training_iter,
        low_mapping_dim=low_mapping_dim, prediction_act=prediction_act, use_grow_prune_prob=use_grow_prune_prob
    )

    contrastive_loss_fn = ContrastiveLoss(m=margin) # if I am using tanh, range is between -1 and 1.
    val_contrastive_loss_fn = ContrastiveLoss(m=margin, reduction='sum')

    train_losses_epoch, val_losses_epoch, test_df, model_state_dicts = full_train(
        DrLIM_model, train_pair_dataloader, val_pair_dataloader, test_pair_dataloader,
        model_training_output_dir=output_dir,
        override=True,
        learning_rate=learning_rate, 
        loss_fn=contrastive_loss_fn,
        val_loss_fn=val_contrastive_loss_fn,
        plot=False, verbose=False,
        test_loop_func=constrative_test_loop,
        margin=contrastive_loss_fn.m,
        args_expand=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--subset_fraction", type=float, default=0.5, help="Fraction of dataset to train on")
    parser.add_argument("--selected_labels", default="49", help="string of single digits to classify (e.g. '4,9')")
    parser.add_argument("--validation_ratio", type=int, default=6, help="1/N for validation")
    parser.add_argument("--init_density", type=float, default=0.5, help="Initial density of the MLP")
    parser.add_argument("--num_training_iter", type=int, default=100, help="Number of epochs of training. Sets pruning rate")
    parser.add_argument("--low_mapping_dim", type=int, default=2, help="Number of dimensions for final mapping")
    parser.add_argument("--prediction_act_type", type=str, default="linear", help="linear or tanh")
    parser.add_argument("--margin", type=float, default=5, help="Margin size for contrastive loss")
    parser.add_argument("--use_grow_prune_prob", type=bool, default=False, help="Whether to prune/grow or not")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="SGD rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--job_id", type=str, default=None, help="sbatch script job id (%j)")
    parser.add_argument("--seed", type=int, default=4, help="Seed for selecting training data")

    args = parser.parse_args()

    selected_labels = [int(i) for i in args.selected_labels]

    parameters_abbr = {
        "bs": args.batch_size, 
        "sf": args.subset_fraction, 
        "sl": args.selected_labels, 
        "vr": args.validation_ratio, 
        "id": args.init_density, 
        "nti": args.num_training_iter, 
        "lmd": args.low_mapping_dim, 
        "pat": args.prediction_act_type, 
        "m": args.margin, 
        "ugpp": args.use_grow_prune_prob, 
        "lr": args.learning_rate, 
        "s": args.seed, 
    }

    parameters_dir = "_".join([f"{abbr}-{data}" for abbr, data in parameters_abbr.items()])
    full_output_dir = f"{args.output_dir}/{parameters_dir}_sbatch-{args.job_id}"

    main(
        batch_size=args.batch_size,
        subset_fraction=args.subset_fraction,
        selected_labels=selected_labels,
        validation_ratio=args.validation_ratio,
        init_density=args.init_density,
        num_training_iter=args.num_training_iter,
        low_mapping_dim=args.low_mapping_dim,
        prediction_act_type=args.prediction_act_type,
        margin=args.margin,
        use_grow_prune_prob=args.use_grow_prune_prob,
        learning_rate=args.learning_rate,
        output_dir=full_output_dir,
        seed=args.seed,
    )