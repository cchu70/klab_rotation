import numpy as np
import pandas as pd
from sklearn import metrics
import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import os
import time
import gzip
import copy

def save_model_attr(model, attr_fn):
    attr_dict = model.get_attribute_dict()
    # attr_dict = {
    #     'total_size': model.total_size,
    #     'gamma': model.gamma,
    #     'grow_prob': model.grow_prob,
    #     'grow_prune_history': model.grow_prune_history,
    #     'synapse_count_history': model.synapse_count_history,
    #     'grow_prob_history': model.grow_prob_history,
    #     'num_training_iter': model.num_training_iter,
    # }
    # trace is too large
    with open(attr_fn, 'wb') as fp:
        pickle.dump(attr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def train_loop(
    dataloader, model, loss_fn, optimizer, verbose_print, val_dataloader=None, val_batch_freq=10, 
    val_loss_fn = nn.CrossEntropyLoss(reduction='sum'), args_expand=False,
):
    size = len(dataloader.dataset)
    # set model to training mode
    # generator = torch.Generator().manual_seed(42)
    val_loss = 0

    train_losses = dict()
    val_losses = dict()
    
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        pred = model(X)

        # with torch.autograd.detect_anomaly():
        if args_expand:
            loss = loss_fn(*pred, y)
        else:
            loss = loss_fn(pred, y)
        train_losses[batch] = loss.item()
    
        # backprop
        loss.backward()

        # Check gradients
        # print(model.layers[1].W.grad)
        # print(train_losses[batch])
        # assert (model.layers[1].W.grad.view(-1)[torch.argwhere(model.layers[1].A.view(-1) == 0.0)] == 0.0).all()
        
        optimizer.step()
        optimizer.zero_grad() #reset gradients of model parameters for each batch to avoid double counting

        if batch % val_batch_freq == 0:
            loss, current = loss.item(), batch * len(X) + len(X)
            verbose_print(f"loss: {loss:7f} [{current:>5d}/{size:>5d}]")

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                num_data = 0.0
                for val_batch, (X_val, y_val) in enumerate(val_dataloader):
                    pred = model(X_val)
                    if args_expand:
                        val_loss += val_loss_fn(*pred, y_val).item()
                    else:
                        val_loss += val_loss_fn(pred, y_val).item()
                    num_data += len(y_val)
            val_losses[batch] = val_loss / num_data
            verbose_print(f"Val loss: {val_loss:7f}")

    return train_losses, val_losses, model.state_dict()

def test_loop(
    dataloader, model, loss_fn, verbose_print,
    correct_func=lambda pred, y, *args, **kwargs: (pred.argmax(1) == y).type(torch.float).sum().item(), 
    args_expand=False
):
    """
    correct_func: takes prediction as first input and expected value as second input
    """
    # set to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad(): # no gradients computed
        for X, y in dataloader:
            pred = model(X)
            if args_expand:
                test_loss += loss_fn(*pred, y).item()
                correct += correct_func(pred, y)
            else:
                test_loss += loss_fn(pred, y).item()
                correct += correct_func(pred, y) #(pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    
    verbose_print(f"Test Error \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>3f}\n")
    return correct, test_loss

def full_train(
    model,
    train_dataloader, val_dataloader, test_dataloader,
    model_training_output_dir,
    override=False,
    learning_rate = 1e-2, # how much to update model parameters at each epoch. Speed of learning
    loss_fn = nn.CrossEntropyLoss(),
    val_loss_fn = nn.CrossEntropyLoss(reduction='sum'),
    model_update_params=True,
    plot=False,
    verbose=True,
    args_expand=False,
    test_loop_func=test_loop,
    **test_loop_kwargs,
):
    verbose_print = print if verbose else lambda *x: None

    if not os.path.exists(model_training_output_dir):
        os.makedirs(model_training_output_dir)
        print(f"Created new directory {model_training_output_dir}")
    elif not override:
        raise ValueError(f"{model_training_output_dir} already exists. Set override=True or provide a different path.")
    else:
        print(f"{model_training_output_dir} already exists and override is True. Interrupt before training ends to prevent overwriting.")
    
    epochs = model.num_training_iter
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # make sure model is the right instance
    test_df = pd.DataFrame(index=np.arange(epochs), columns=['test_accuracy', 'test_loss'])
    test_df.index.name = 'epoch'
    train_losses_epoch = dict()
    val_losses_epoch = dict()

    if plot: plot_model_state(model)
    model_state_dicts = {}
    for i, t in tqdm.tqdm(enumerate(range(epochs)), desc='Epochs', total=epochs):
        train_losses, val_losses, model_state_dict = train_loop(
            train_dataloader, model, loss_fn, optimizer, verbose_print, 
            val_dataloader, val_loss_fn=val_loss_fn, args_expand=args_expand
        )
        train_losses_epoch[i] = train_losses
        val_losses_epoch[i] = val_losses
        model_state_dicts[i] = copy.deepcopy(model_state_dict)

        model.eval()

        if model_update_params:
            decision = model.update_params()

        if plot: 
            verbose_print(f"Decision: {decision}")
            plot_model_state(model)
        
        test_err, test_loss = test_loop_func(test_dataloader, model, val_loss_fn, verbose_print, **test_loop_kwargs)
        test_df.loc[i] = [test_err, test_loss]
    
    verbose_print("done!")

    verbose_print("Save training data")
    save_training_data(model_training_output_dir, model, train_losses_epoch, val_losses_epoch, test_df, model_state_dicts)


    return train_losses_epoch, val_losses_epoch, test_df, model_state_dicts

def save_training_data(output_dir, model, train_losses_epoch, val_losses_epoch, test_df, model_state_dicts):
    model_attr_fn = f"{output_dir}/model_attr.pkl"
    save_model_attr(model, model_attr_fn)

    test_err_loss_fn = f"{output_dir}/test_err_loss.tsv"
    test_df.to_csv(test_err_loss_fn, sep='\t')

    stack_training_losses_df, stack_val_losses_df = format_training_outputs(train_losses_epoch, val_losses_epoch)

    stack_training_losses_fn = f"{output_dir}/stack_training_losses.tsv"
    stack_val_losses_fn = f"{output_dir}/stack_val_losses.tsv"
    
    stack_training_losses_df.to_csv(stack_training_losses_fn, sep='\t')
    stack_val_losses_df.to_csv(stack_val_losses_fn, sep='\t')

    model_state_dicts_pkl = f"{output_dir}/model_state_dicts.pkl.gz"
    # with open(model_state_dicts_pkl, 'wb') as fh:
    #     pickle.dump(model_state_dicts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(model_state_dicts_pkl, 'wb') as fh:
        pickle.dump(model_state_dicts, fh, protocol=pickle.HIGHEST_PROTOCOL)

def plot_model_state(model, layer_idx=1):
    with torch.no_grad():
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(model.layers[layer_idx].A.detach().numpy(), cmap='gray')
        axes[1].imshow(model.layers[layer_idx].W.detach().numpy(), cmap='seismic', vmax=1.0, vmin=-1.0)
        plt.show()
    
        # plot weight distribution
        A_mask = model.layers[layer_idx].A.view(-1).detach().numpy()
        A_mask= A_mask.astype(int) == 1.0
        plt.hist(
            model.layers[layer_idx].W.view(-1).detach().numpy()[A_mask], bins=20
        )
        plt.title("Active weights")
        plt.ylabel("Num weights")
        plt.show()
    
        plt.hist(
            np.abs(model.layers[layer_idx].W.view(-1).detach().numpy())[A_mask], bins=20
        )
        plt.title("Absolute active weights")
        plt.ylabel("Num weights")
        plt.show()

def format_training_outputs(train_losses_epoch, val_losses_epoch):
    training_losses_df = pd.DataFrame(train_losses_epoch).T
    training_losses_df.columns.name = 'batch'
    training_losses_df.index.name = 'epoch'
    
    val_losses_df = pd.DataFrame(val_losses_epoch).T
    val_losses_df.columns.name = 'batch'
    val_losses_df.index.name = 'epoch'
    
    stack_training_losses_df = training_losses_df.stack().reset_index().rename(columns={0: 'loss'})
    stack_training_losses_df.index = stack_training_losses_df['epoch'].astype(str) + '-' + stack_training_losses_df['batch'].astype(str)
    stack_training_losses_df.index.name = 'index'
    stack_val_losses_df = val_losses_df.stack().reset_index().rename(columns={0: 'loss'})
    stack_val_losses_df.index = stack_val_losses_df['epoch'].astype(str) + '-' + stack_val_losses_df['batch'].astype(str)
    stack_val_losses_df.index.name = 'index'

    return stack_training_losses_df, stack_val_losses_df

