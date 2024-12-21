from simple_pruning_growth_model import PruneGrowNetwork
from training_testing_loop import full_train
from load_MNIST import load_MNIST
from torch import nn
import argparse
        

def main(
    gamma=0.1, batch_size=32, subset_fraction=0.5, selected_labels=[4,9], validation_ratio=6,
    init_density=0.5, num_training_iter=100, use_grow_prune_prob=False, 
    learning_rate=1e-3, 
    output_dir=None,
    seed=42,
):

    if output_dir is None:
        raise ValueError("Set an output directory")
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = load_MNIST(
        root='./data', subset_frac=subset_fraction, 
        batch_size=batch_size, validation_ratio=validation_ratio, seed=seed
    )

    model = PruneGrowNetwork(gamma=gamma, init_density=init_density, num_training_iter=num_training_iter, use_grow_prune_prob=use_grow_prune_prob)


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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--subset_fraction", type=float, default=0.5, help="Fraction of dataset to train on")
    parser.add_argument("--validation_ratio", type=int, default=6, help="1/N for validation")
    parser.add_argument("--init_density", type=float, default=0.5, help="Initial density of the MLP")
    parser.add_argument("--num_training_iter", type=int, default=100, help="Number of epochs of training. Sets pruning rate")
    parser.add_argument("--use_grow_prune_prob", action=argparse.BooleanOptionalAction, default=False, help="Whether to prune/grow or not")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="SGD rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--job_id", type=str, default=None, help="sbatch script job id (%j)")
    parser.add_argument("--desc", type=str, default=None, help="Description of the run, no spaces or special characters except _ and -")
    parser.add_argument("--seed", type=int, default=4, help="Seed for selecting training data")

    args = parser.parse_args()

    parameters_abbr = {
        "bs": args.batch_size, 
        "sf": args.subset_fraction, 
        "vr": args.validation_ratio, 
        "id": args.init_density, 
        "nti": args.num_training_iter, 
        "ugpp": args.use_grow_prune_prob, 
        "lr": args.learning_rate, 
        "s": args.seed, 
    }

    parameters_dir = "_".join([f"{abbr}-{data}" for abbr, data in parameters_abbr.items()])
    full_output_dir = f"{args.output_dir}/{args.desc}/sbatch-{args.job_id}_{parameters_dir}"

    main(
        batch_size=args.batch_size,
        subset_fraction=args.subset_fraction,
        validation_ratio=args.validation_ratio,
        init_density=args.init_density,
        num_training_iter=args.num_training_iter,
        use_grow_prune_prob=args.use_grow_prune_prob,
        learning_rate=args.learning_rate,
        output_dir=full_output_dir,
        seed=args.seed,
    )