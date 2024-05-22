import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""## Part 1: Graph Data Preparation"""

data = pd.read_excel("./data/Node data_clean.xlsx", index_col="Date")


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


"""## Model architecture"""

from typing import Union, Callable, Optional


class FullyConnected(nn.Module):
    r"""Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
    """

    def __init__(self,
        input_dims: int,
        output_dims: int,
        kernel_size: Union[tuple, list] = (1, 1),
        stride: Union[tuple, list] = (1, 1),
        use_bias: bool = True,
        activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.tanh,
    ):
        super(FullyConnected, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self._batch_norm = nn.BatchNorm2d(output_dims)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """ Arg types:
            * **X** - Input tensor: (batch_size, num_his, num_nodes, input_dims).
            Return types:
            * **X** - Output tensor: (batch_size, num_his, num_nodes, output_dims).
        """
        X = X.permute(0, 3, 2, 1) # [B, D, N, S]
        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1) # [B, S, N, D]


class TemporalAttention(nn.Module):
    r""" Args:
        K (int) : Number of attention heads.
        D (int) : number of input and output channels
        mask (bool): Whether to mask attention score."""

    def __init__(self, K: int, D: int, mask: bool):
        super(TemporalAttention, self).__init__()
        self._d =  D // K
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(
            input_dims=D, output_dims=D, activation=F.relu)
        self._fully_connected_k = FullyConnected(
            input_dims=D, output_dims=D, activation=F.relu)
        self._fully_connected_v = FullyConnected(
            input_dims=D, output_dims=D, activation=F.relu)
        self._fully_connected = FullyConnected(
            input_dims=D, output_dims=D, activation=F.relu)

    def forward(
        self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Arg types:
            * **X** - Input sequence, (batch_size, num_step, num_nodes, K*d).
           Return types:
            * **X** - Temporal attention scores: (batch_size, num_step, num_nodes, K*d)."""

        batch_size = X.shape[0]
        query = self._fully_connected_q(X) # [B, S, N, K*d]
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3) # [kB, N, S, d]
        key = key.permute(0, 2, 3, 1) # [kB, N, d, S]
        value = value.permute(0, 2, 1, 3) # [kB, N, S, d]
        attention = torch.matmul(query, key) # [kB, N, S, S]
        attention /= self._d ** 0.5
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step).to(X.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            mask = mask.to(torch.bool)
            condition = torch.FloatTensor([-(2 ** 15) + 1]).to(X.device)
            attention = torch.where(mask, attention, condition) #(32) must match the size of tensor b (128)
        attention = F.softmax(attention, dim=-1) # [kB, N, S, S]
        X = torch.matmul(attention, value) # [kB, N, S, d]
        X = X.permute(0, 2, 1, 3) # [kB, S, N, d]
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1) # [B, N, S, d*k]
        X = self._fully_connected(X)
        del query, key, value, attention
        return X

class Linear(nn.Module):
    r""" Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

           Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixProp(nn.Module):
    r"""Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()

        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha
        self._mlp = Linear((gdep + 1) * c_in, c_out)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

           Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            H = self._alpha * X + (1 - self._alpha) * torch.einsum(
                "ncwl,vw->ncvl", (H, A)
            )
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    r"""Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

           Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        X = torch.cat(X, dim=1)
        return X

class GraphConstructor(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(
        self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)

        self._k = k
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * **FE** (Pytorch Float Tensor, optional) - Static feature, default None.
        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """

        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A


class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("_weight", None)
            self.register_parameter("_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            nn.init.ones_(self._weight)
            nn.init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        """Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.

          Return types:
            * **X** (PyTorch Float Tensor) - Output tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
        """
        if self._elementwise_affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps)
        else:
            return F.layer_norm(X, tuple(X.shape[1:]), self._weight, self._bias, self._eps)


class MTGNNLayer(nn.Module):

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        propalpha: float,
    ):
        super(MTGNNLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true

        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._filter_attn = TemporalAttention(K= heads, D=conv_channels, mask= True)

        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._gate_attn = TemporalAttention(K= heads, D=conv_channels, mask= True)

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        if gcn_true:
            self._mixprop_conv1 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

            self._mixprop_conv2 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN layer.

        Arg types:
            * **X**  - Input feature tensor: (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** - Input feature tensor for skip connection: (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor or None) - Predefined adjacency matrix.
            * **idx** - Input indices.
            * **training** (bool) - Whether in traning mode.
        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, seq_len, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = self._filter_attn(X_filter.permute(0, 3, 2, 1))
        X_filter = torch.tanh(X_filter.permute(0,3,2,1))
        X_gate = self._gate_conv(X)
        X_gate = self._filter_attn(X_gate.permute(0, 3, 2, 1))
        X_gate = torch.sigmoid(X_gate.permute(0,3,2,1))
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class MTGNN_Att(nn.Module):

    def __init__(
        self,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        dropout: float,
        subgraph_size: int,
        node_dim: int,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        layers: int,
        propalpha: float,
        tanhalpha: float,
        layer_norm_affline: bool,
        xd: Optional[int] = None,
    ):
        super(MTGNN_Att, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)

        self._mtgnn_layers = nn.ModuleList()

        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, xd=xd
        )

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propalpha=propalpha,
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )

        self._reset_parameters()

    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):

        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(
        self,
        X_in: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** - Input sequence: (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** - Predefined adjacency matrix, default None.
            * **idx** - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE**  - Static feature, default None.

        Return types:
            * **X** - Output sequence for prediction: (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(
                X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
            )

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(
                    X, X_skip, A_tilde, self._idx.to(X_in.device), self.training
                )
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training) # X_skip [B, C_skip, N, 1]

        X_skip = self._skip_conv_E(X) + X_skip # X_skip [B, C_skip, N, T_out]
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        # return X
        return X , A_tilde

