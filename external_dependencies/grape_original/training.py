import numpy as np
import torch
import torch.nn.functional as F

from .gnn_model import get_gnn
from .prediction_model import MLPNet
from .utils import objectview


def build_optimizer(args, params):
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = torch.optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    return scheduler, optimizer


def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask


def mask_edge(edge_index, edge_attr, mask, remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.0
    return edge_index, edge_attr


def train_gnn_mdi(data, args, device=torch.device("cpu"), return_filled_X=False):
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == "":
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split("_")))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    output_dim = 1
    impute_model = MLPNet(
        input_dim,
        output_dim,
        hidden_layer_sizes=impute_hiddens,
        hidden_activation=args.impute_activation,
        dropout=args.dropout,
    ).to(device)

    trainable_parameters = list(model.parameters()) + list(impute_model.parameters())
    scheduler, opt = build_optimizer(args, trainable_parameters)

    x = data.x.clone().detach().to(device)
    train_edge_index, train_edge_attr, train_labels = data.train_edge_index.clone().detach().to(device), data.train_edge_attr.clone().detach().to(device), data.train_labels.clone().detach().to(device)
    test_input_edge_index, test_input_edge_attr = train_edge_index, train_edge_attr
    test_edge_index, test_edge_attr, test_labels = data.test_edge_index.clone().detach().to(device), data.test_edge_attr.clone().detach().to(device), data.test_labels.clone().detach().to(device)

    best_rmse = float("inf")
    best_state = None
    best_impute_state = None
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        pred_train = pred[: int(train_edge_attr.shape[0] / 2), 0]
        if args.loss_mode == 1:
            pred_train[known_mask] = train_labels[known_mask]
        loss = F.mse_loss(pred_train, train_labels)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step(epoch)

        model.eval()
        impute_model.eval()
        with torch.no_grad():
            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            pred_test = pred[: int(test_edge_attr.shape[0] / 2), 0]
            mse = F.mse_loss(pred_test, test_labels)
            test_rmse = np.sqrt(mse.item())

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_impute_state = {k: v.detach().cpu().clone() for k, v in impute_model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= getattr(args, "early_stopping_patience", 20):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_impute_state is not None:
        impute_model.load_state_dict(best_impute_state)

    if return_filled_X:
        model.eval()
        impute_model.eval()
        with torch.no_grad():
            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            pred_test = pred[: int(test_edge_attr.shape[0] / 2), 0]
            filled_X = data.df_X.values.reshape(-1)
            filled_X[~data.train_edge_mask] = pred_test.detach().cpu().numpy()
            return filled_X.reshape(data.df_X.shape), impute_model, model

    return None, impute_model, model


def out_of_sample_test_gnn_mdi(data_test, impute_model, model, device, return_filled_X=True):
    x = data_test.x.clone().detach().to(device)
    model.eval()
    impute_model.eval()

    with torch.no_grad():
        all_train_edge_index = data_test.train_edge_index.clone().detach().to(device)
        all_train_edge_attr = data_test.train_edge_attr.clone().detach().to(device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data_test.test_edge_index.clone().detach().to(device)

        x_embd = model(x, test_input_edge_attr, test_input_edge_index)
        pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
        pred_test = pred[: int(data_test.test_edge_attr.shape[0] / 2), 0]

        filled_X = data_test.df_X.values.reshape(-1)
        filled_X[~data_test.train_edge_mask] = pred_test.detach().cpu().numpy()
        return filled_X.reshape(data_test.df_X.shape)
