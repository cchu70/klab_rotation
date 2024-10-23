import numpy as np
import pandas as pd
from sklearn import metrics
import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle

def save_model_attr(model, attr_fn):
    attr_dict = {
        'total_size': model.total_size,
        'gamma': model.gamma,
        'grow_prob': model.grow_prob,
        'grow_prune_history': model.grow_prune_history,
        'synapse_count_history': model.synapse_count_history,
        'grow_prob_history': model.grow_prob_history,
        'num_training_iter': model.num_training_iter,
    }
    # trace is too large
    with open(attr_fn, 'wb') as fp:
        pickle.dump(attr_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def train_loop(
    dataloader, model, loss_fn, optimizer, verbose_print, val_dataloader=None, val_batch_freq=10
):
    size = len(dataloader.dataset)
    # set model to training mode
    # generator = torch.Generator().manual_seed(42)
    val_loss = 0

    train_losses = dict()
    val_losses = dict()

    val_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        train_losses[batch] = loss.item()
        
        # backprop
        loss.backward()
        # Check gradients
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
                for val_batch, (X_val, y_val) in enumerate(val_dataloader):
                    pred = model(X_val)
                    val_loss += val_loss_fn(pred, y_val).item()
            val_losses[batch] = val_loss / len(val_dataloader.dataset)
            verbose_print(f"Val loss: {val_loss:7f}")

    return train_losses, val_losses

def test_loop(dataloader, model, loss_fn, verbose_print):
    # set to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad(): # no gradients computed
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    
    verbose_print(f"Test Error \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>3f}\n")
    return correct, test_loss

def full_train(
    model,
    train_dataloader, val_dataloader, test_dataloader,
    learning_rate = 1e-2, # how much to update model parameters at each epoch. Speed of learning
    loss_fn = nn.CrossEntropyLoss(),
    model_update_params=True,
    plot=False,
    verbose=True,
):
    verbose_print = print if verbose else lambda *x: None
    
    epochs = model.num_training_iter
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # make sure model is the right instance
    test_df = pd.DataFrame(index=np.arange(epochs), columns=['test_err', 'test_loss'])
    test_df.index.name = 'epoch'
    train_losses_epoch = dict()
    val_losses_epoch = dict()

    if plot: plot_model_state(model)
    for i, t in tqdm.tqdm(enumerate(range(epochs)), desc='Epochs', total=epochs):
        train_losses, val_losses = train_loop(train_dataloader, model, loss_fn, optimizer, verbose_print, val_dataloader)
        train_losses_epoch[i] = train_losses
        val_losses_epoch[i] = val_losses

        if model_update_params:
            model.update_params()

        if plot: plot_model_state(model)
        
        test_err, test_loss = test_loop(test_dataloader, model, loss_fn, verbose_print)
        test_df.loc[i] = [test_err, test_loss]
    
    print("done!")
    return train_losses_epoch, val_losses_epoch, test_df

def plot_model_state(model, layer_idx=1):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(model.layers[layer_idx].A.detach().numpy(), cmap='gray')
    axes[1].imshow(model.layers[layer_idx].W.detach().numpy(), cmap='seismic', vmax=1.0, vmin=-1.0)
    plt.show()

