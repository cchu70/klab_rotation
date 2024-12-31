from simple_pruning_growth_model import ContrastiveLoss, constrative_test_loop
from CNN_pruning import MiniAlexNet, RandomPruneNet, ActivityPruneNet
from training_testing_loop import full_train
from load_MNIST import load_MNIST
from load_CIFAR10 import get_train_valid_loader, get_test_loader
from torch import nn
import argparse
        

def main(
    dataset, batch_size=32, subset_fraction=0.5, validation_ratio=6,
    num_training_iter=100, num_classes=10, prune_model_type="None",  
    learning_rate=1e-3, gamma=0.1,
    output_dir=None,
    seed=42,
    in_channels=1,
):
    # get dataloaders
    if dataset == 'MNIST':
        train_dataloader, val_dataloader, test_dataloader = load_MNIST(
            root='./data', subset_frac=subset_fraction, 
            batch_size=batch_size, validation_ratio=validation_ratio, seed=seed
        )
        assert in_channels == 1
    elif dataset == 'CIFAR10':
        train_dataloader, val_dataloader = get_train_valid_loader(
            data_dir='./data/', batch_size=batch_size, random_seed=seed, 
            augment=False, download=True, subsample_frac=subset_fraction
        )
        test_dataloader = get_test_loader(data_dir='./data', batch_size=batch_size)
        assert in_channels == 3

    model_args = dict(
         num_training_iter=num_training_iter, 
         num_classes=num_classes, 
         gamma=gamma, 
         verbose=False, 
         random_seed=seed,
         in_channels=in_channels,
    )
    
    if prune_model_type=="NoPrune":
        model = MiniAlexNet(**model_args)
    elif prune_model_type=="Activity":
        model = ActivityPruneNet(**model_args)
    elif prune_model_type=="Random":
        model = RandomPruneNet(**model_args)
    else:
        raise ValueError(f"prune_model={prune_model_type} is not valid. Options are ['NoPrune', 'Activity', 'Random']")

    if output_dir is None:
        raise ValueError("Set an output directory")

    loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    train_losses_epoch, val_losses_epoch, test_df, model_state_dicts = full_train(
        model, train_dataloader, val_dataloader, test_dataloader,
        model_training_output_dir=output_dir,
        override=True,
        learning_rate=learning_rate, 
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        plot=False, verbose=False,
        args_expand=False,
        split_model_states=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='CIFAR10', help="CIFAR10 (3 in channels) or MNIST (1 in channel)")
    parser.add_argument("--in_channels", type=int, default=3, help="number of channels in the images of the dataset. (see --dataset)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--subset_fraction", type=float, default=0.5, help="Fraction of dataset to train on")
    parser.add_argument("--selected_labels", default="49", help="string of single digits to classify (e.g. '4,9')")
    parser.add_argument("--validation_ratio", type=int, default=6, help="1/N for validation")
    
    # CNN model to train
    parser.add_argument("--prune_model_type", type=str, default="None", help="Activity, Random, or NoPrune")
    parser.add_argument("--gamma", type=float, default=0.1, help="Pruning amount")
    parser.add_argument("--num_training_iter", type=int, default=100, help="Number of epochs of training. Sets pruning rate")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for final classification")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="SGD rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--job_id", type=str, default=None, help="sbatch script job id (pct j in the comments in sbatch script)")
    parser.add_argument("--desc", type=str, default=None, help="Description of the run, no spaces or special characters except _ and -")
    parser.add_argument("--seed", type=int, default=4, help="Seed for selecting training data")

    args = parser.parse_args()

    selected_labels = [int(i) for i in args.selected_labels]

    parameters_abbr = {
        "ds": args.dataset,
        "ic": args.in_channels,
        "bs": args.batch_size, 
        "sf": args.subset_fraction, 
        "sl": args.selected_labels, 
        "vr": args.validation_ratio, 
        "nti": args.num_training_iter, 
        "lmd": args.num_classes, 
        "m": args.margin, 
        "pmt": args.prune_model_type,
        "g": args.gamma, 
        "lr": args.learning_rate, 
        "s": args.seed, 
    }

    parameters_dir = "_".join([f"{abbr}-{data}" for abbr, data in parameters_abbr.items()])
    full_output_dir = f"{args.output_dir}/{args.desc}/sbatch-{args.job_id}_{parameters_dir}"

    main(
        dataset=args.dataset,
        batch_size=args.batch_size,
        subset_fraction=args.subset_fraction,
        validation_ratio=args.validation_ratio,
        num_training_iter=args.num_training_iter,
        num_classes=args.num_classes,
        gamma=args.gamma,
        prune_model_type=args.prune_model_type,
        learning_rate=args.learning_rate,
        output_dir=full_output_dir,
        seed=args.seed,
    )