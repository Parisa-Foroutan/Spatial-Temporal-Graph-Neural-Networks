import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelArchitecture import MTGNN_Att
from helperFunctions import *

torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""### Define Parameters ans Split Data"""

in_dim = 1
out_dim = 1
n_out_seq = 1
num_epochs = 40
clip = 5

# model parameters
kernel_set = [2, 3, 6, 7]
gcn_depth = 2

## valid_acc = 0.80346
n_in_seq = 20 # number of historical time steps to consider
batch_size = 32
learning_rate = 0.005
heads = 4
layers = 2 # number of MTGNNLayer
node_emb_dim = 32
out = 32
skip = 16
conv_res = 64
subgraph_size = 15
reg_penalty = 0.0001
batch_size = 16
dropout= 0.5

"""## Prediction and evaluation on test set"""

best_model = MTGNN_Att(gcn_true= True, build_adj= True, gcn_depth= gcn_depth,
              num_nodes= num_nodes, kernel_set= kernel_set, kernel_size=7,
              dropout= dropout, subgraph_size= subgraph_size, node_dim= node_emb_dim,
              dilation_exponential=2, conv_channels= conv_res,
              residual_channels = conv_res, skip_channels= skip, end_channels= out,
              seq_length= n_in_seq, in_dim= 1, out_dim= n_out_seq, layers= layers,
              propalpha= 0.05, tanhalpha= 3, layer_norm_affline= True, xd= None).to(device)

best_model.load_state_dict(torch.load(model_save_path))

# evaluate model on test set
test_acc , test_loss = evaluate_model(best_model, loss, test_iter, device)



############ Predictions ############
y, y_pred = get_predictions(best_model, test_iter, device)

############ Save the Predictions #############
np.savetxt(real_save_path, y, delimiter=',')
np.savetxt(pred_save_path, y_pred, delimiter=',')

############ Evaluation Metrics ############
target_node_idx = ['WTI', 'Brent', 'Silver', 'Gold']
metric_names = ["Accuracy", "F1-score"]
metric_values = []
for i, target in enumerate(target_node_idx):
  real = y[:,i]
  preds = y_pred[:,i]
  acc = (real==preds).sum()/len(y)
  f1 = f1_score(real, preds)}

  metric_values.append(np.array([acc,f1]))

test_performance = pd.DataFrame(metric_values, index= target_node_idx, columns = metric_names)
test_performance.head() 
