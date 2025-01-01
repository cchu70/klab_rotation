from simple_pruning_growth_model import ContrastiveLoss, contrastive_test_loop
from CNN_pruning import DrLIMMiniAlexNet, DrLIMRandomPruneNet, DrLIMActivityPruneNet
from training_testing_loop import full_train
from load_MNIST import get_mnist_pairs_loader
from torch import nn
import argparse
        

def main(
    batch_size=32, subset_fraction=0.5, selected_labels=[4,9], validation_ratio=6,
    num_training_iter=100, num_pretraining=100, low_mapping_dim=2, prune_model_type="None", margin=5, 
    learning_rate=1e-3, gamma=0.1,
    output_dir=None,
    seed=42,
    in_channels=1,
):
    model_args = dict(
         num_training_iter=num_training_iter, 
         num_pretraining=num_pretraining,
         num_classes=low_mapping_dim, 
         gamma=gamma, 
         verbose=False, 
         random_seed=seed,
         in_channels=in_channels,
    )
    
    if prune_model_type=="NoPrune":
        model = DrLIMMiniAlexNet(**model_args)
    elif prune_model_type=="Activity":
        model = DrLIMActivityPruneNet(**model_args)
    elif prune_model_type=="Random":
        model = DrLIMRandomPruneNet(**model_args)
    else:
        raise ValueError(f"prune_model={prune_model_type} is not valid. Options are []'None', 'Activity', 'Random']")

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

    contrastive_loss_fn = ContrastiveLoss(m=margin) # if I am using tanh, range is between -1 and 1.
    val_contrastive_loss_fn = ContrastiveLoss(m=margin, reduction='sum')

    train_losses_epoch, val_losses_epoch, test_df, model_state_dicts = full_train(
        model, train_pair_dataloader, val_pair_dataloader, test_pair_dataloader,
        model_training_output_dir=output_dir,
        override=True,
        learning_rate=learning_rate, 
        loss_fn=contrastive_loss_fn,
        val_loss_fn=val_contrastive_loss_fn,
        plot=False, verbose=False,
        args_expand=True,
        split_model_states=True,
        test_loop_func=contrastive_test_loop,
        margin=contrastive_loss_fn.m,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--subset_fraction", type=float, default=0.5, help="Fraction of dataset to train on")
    parser.add_argument("--selected_labels", default="49", help="string of single digits to classify (e.g. '4,9')")
    parser.add_argument("--validation_ratio", type=int, default=6, help="1/N for validation")
    
    # CNN model to train
    parser.add_argument("--prune_model_type", type=str, default="None", help="Activity, Random, or NoPrune")
    parser.add_argument("--gamma", type=float, default=0.1, help="Pruning amount")
    parser.add_argument("--num_training_iter", type=int, default=100, help="Number of epochs of training. Sets pruning rate")
    parser.add_argument("--num_pretraining", type=int, default=None, help="Number of epochs of pretraining with no pruning")
    parser.add_argument("--low_mapping_dim", type=int, default=2, help="Number of dimensions for final mapping")

    # Loss parameter
    parser.add_argument("--margin", type=float, default=5, help="Margin size for contrastive loss")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="SGD rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--job_id", type=str, default=None, help="sbatch script job id (pct j in the comments in sbatch script)")
    parser.add_argument("--desc", type=str, default=None, help="Description of the run, no spaces or special characters except _ and -")
    parser.add_argument("--seed", type=int, default=4, help="Seed for selecting training data")

    args = parser.parse_args()

    selected_labels = [int(i) for i in args.selected_labels]

    parameters_abbr = {
        "bs": args.batch_size, 
        "sf": args.subset_fraction, 
        "sl": args.selected_labels, 
        "vr": args.validation_ratio, 
        "nti": args.num_training_iter, 
        "pt": args.num_pretraining,
        "lmd": args.low_mapping_dim, 
        "m": args.margin, 
        "pmt": args.prune_model_type,
        "g": args.gamma, 
        "lr": args.learning_rate, 
        "s": args.seed, 
    }

    parameters_dir = "_".join([f"{abbr}-{data}" for abbr, data in parameters_abbr.items()])
    full_output_dir = f"{args.output_dir}/{args.desc}/sbatch-{args.job_id}_{parameters_dir}"

    main(
        batch_size=args.batch_size,
        subset_fraction=args.subset_fraction,
        selected_labels=selected_labels,
        validation_ratio=args.validation_ratio,
        num_training_iter=args.num_training_iter,
        num_pretraining=args.num_pretraining,
        low_mapping_dim=args.low_mapping_dim,
        gamma=args.gamma,
        margin=args.margin,
        prune_model_type=args.prune_model_type,
        learning_rate=args.learning_rate,
        output_dir=full_output_dir,
        seed=args.seed,
    )