import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch_geometric.data import Data


def create_node(df, mode):
    if mode == 0:
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1] * ncol for _i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1:
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol + 1))
        feature_node[np.arange(ncol), feature_ind + 1] = 1
        sample_node = np.zeros((nrow, ncol + 1))
        sample_node[:, 0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    else:
        raise NotImplementedError
    return node


def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col
        edge_end = edge_end + list(n_row + np.arange(n_col))
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)


def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i, j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr


def mask_edge(edge_index, edge_attr, mask, remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.0
    return edge_index, edge_attr


def get_data_fix_mask(df_X, train_mask, node_mode=0, seed=0, df_y=None):
    if df_y is None:
        df_y = pd.DataFrame(np.zeros(df_X.shape[0]))

    if len(df_y.shape) == 1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape) == 2:
        df_y = df_y[0].to_numpy()

    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode)
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    torch.manual_seed(seed)
    train_edge_mask = torch.tensor(train_mask).reshape(-1)
    train_edge_mask = ~train_edge_mask
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr, double_train_edge_mask, True)
    train_labels = train_edge_attr[: int(train_edge_attr.shape[0] / 2), 0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr, ~double_train_edge_mask, True)
    test_labels = test_edge_attr[: int(test_edge_attr.shape[0] / 2), 0]

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        train_edge_mask=train_edge_mask,
        train_labels=train_labels,
        test_edge_index=test_edge_index,
        test_edge_attr=test_edge_attr,
        test_edge_mask=~train_edge_mask,
        test_labels=test_labels,
        df_X=df_X,
        df_y=df_y,
        edge_attr_dim=train_edge_attr.shape[-1],
        user_num=df_X.shape[0],
    )
    return data
