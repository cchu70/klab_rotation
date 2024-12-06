from src.simple_pruning_growth_model import DrLIMPruneGrowNetwork, ContrastiveLoss, constrative_test_loop
from src.training_testing_loop import full_train, save_model_attr, format_training_outputs
from src.load_MNIST import get_mnist_pairs_loader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import argparse
import pickle
import os
def main(seed, batch_size, learning_rate, subset_fraction, init_density, margin, device, output_dir):
    # get dataloaders
    train_pair_dataloader, val_pair_dataloader = get_mnist_pairs_loader(
        batch_size=batch_size, train=True, subset_fraction=subset_fraction, validation_ratio=6, seed=seed, 
        selected_labels=[4, 9], device=device
    )
    test_pair_dataloader = get_mnist_pairs_loader(
        batch_size=batch_size, train=False, subset_fraction=subset_fraction, 
        selected_labels=[4, 9], device=device
    )

    # initialize model
    DrLIM_model = DrLIMPruneGrowNetwork(
        gamma=0.1, init_density=init_density, num_training_iter=100,
        low_mapping_dim=2, prediction_act=lambda x: x, use_grow_prune_prob=False
    )
    DrLIM_model.to(device)

    # initialize loss functions
    contrastive_loss_fn = ContrastiveLoss(m=margin) # if I am using tanh, range is between -1 and 1.
    val_contrastive_loss_fn = ContrastiveLoss(m=margin, reduction='sum')

    # train model
    train_losses_epoch, val_losses_epoch, test_df, model_state_dicts = full_train(
        DrLIM_model, train_pair_dataloader, val_pair_dataloader, test_pair_dataloader,
        learning_rate = learning_rate, 
        loss_fn=contrastive_loss_fn,
        val_loss_fn=val_contrastive_loss_fn,
        plot=False, verbose=False,
        test_loop_func=constrative_test_loop,
        margin=contrastive_loss_fn.m,
        args_expand=True,
    )

    # save model attributes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    attr_fn = f"{output_dir}/model_attr.pkl"
    save_model_attr(DrLIM_model, attr_fn)

    stack_training_losses_df, stack_val_losses_df = format_training_outputs(train_losses_epoch, val_losses_epoch)
    stack_training_losses_fn = f"{output_dir}/stack_training_losses.csv"
    stack_val_losses_fn = f"{output_dir}/stack_val_losses.csv"
    stack_training_losses_df.to_csv(stack_training_losses_fn, index=False)
    stack_val_losses_df.to_csv(stack_val_losses_fn, index=False)
    
    test_df_fn = f"{output_dir}/test_df.csv"
    test_df.to_csv(test_df_fn, index=False)

    model_state_dicts_fn = f"{output_dir}/model_state_dicts.pkl"
    with open(model_state_dicts_fn, 'wb') as fp:
        pickle.dump(model_state_dicts, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--subset_fraction', type=float, default=0.05, help='Fraction of dataset to use')
    parser.add_argument('--init_density', type=float, default=0.5, help='Initial density for network pruning')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for contrastive loss')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    args = parser.parse_args()

    main(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_fraction=args.subset_fraction,
        init_density=args.init_density,
        margin=args.margin,
        device=args.device,
        output_dir=args.output_dir,
    )