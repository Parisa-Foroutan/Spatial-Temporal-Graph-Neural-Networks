## Set Up


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch


torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Helper Functions


def create_sequences(data, n_in_seq, n_out_seq, EMA, device):
    '''
    X shape: [sampel_size, S_in, N, D_in]), y shape: [sample_sized, N]
    '''

    num_nodes = data.shape[1]
    num_obs = len(data) - n_in_seq - n_out_seq
    X = np.zeros([num_obs, n_in_seq, num_nodes, 1]) # 1 input channel or feature
    y = np.zeros([num_obs, num_nodes])

    for i in range(num_obs):
        head = i
        tail = i + n_in_seq
        X[i, :, :, :] = data[head: tail].reshape(n_in_seq, num_nodes, 1)
        for j in range(num_nodes): # this model predicts for all nodes
          if (data[tail + n_out_seq - 1, j] - EMA[tail + n_out_seq - 1,j] > 0):
        #   if (data[tail + n_out_seq - 1, j] - data[tail + n_out_seq - 2,j] >= 0):
            y[i,j] = 1
          else:
            y[i,j] = 0

    return torch.Tensor(X).to(device), torch.Tensor(y).to(device)

def evaluate_model(model, loss, data_iter, device):
  model.eval()
  l_sum, n, correct = 0.0, 0, 0
  preds = []
  real = []
  with torch.no_grad():
      for x, y in data_iter:
          x = x.permute(0, 3, 2, 1).to(device)
          y = y.to(device)

          y_pred, _ = model(x)
          y_pred = y_pred.view(len(x),-1)
          l = loss(y_pred, y)
          l_sum += l.item() * y.shape[0]
          n += y.shape[0]
        #   y_pred_binary = torch.round(torch.sigmoid(y_pred))
          y_pred_binary = (y_pred > 0).float()
          preds.append(y_pred_binary)
          real.append(y)

      preds = torch.cat(preds, dim=0)
      real = torch.cat(real, dim=0)
      correct += (preds == real).sum().item()
      return correct/(n*real.shape[1]) , l_sum/n

def get_predictions(model, pred_iter, device=device):
    model.eval()
    real = []
    preds = []
    with torch.no_grad():
      for x , y in pred_iter:
        # Input shape: (B, S, N, D) --> (B, D, num_nodes, S)
        x = x.permute(0, 3, 2, 1).to(device)
        num_nodes = y.shape[1]
        y = y.cpu().numpy().reshape(-1, num_nodes)
        y_pred_logit, _ = model(x.to(device))
        y_pred_logit = y_pred_logit.view(len(x), -1).cpu()
        # y_pred = torch.round(torch.sigmoid(y_pred_logit))
        y_pred = (y_pred_logit > 0).float()
        y_pred = y_pred.reshape(-1, num_nodes)

        real.append(y)
        preds.append(y_pred)

      return np.concatenate(real, axis=0), np.concatenate(preds, axis=0)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    return total_params

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(val_losses, label = 'validation loss')
    plt.plot(train_losses, label = 'Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
