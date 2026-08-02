import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.conv import MessagePassing
try:
    from torch_scatter import scatter_add
except (ImportError, OSError):
    # torch_scatter wheels are tightly coupled to a PyTorch ABI.  Native
    # scatter_add_ is mathematically identical for the one-dimensional degree
    # accumulation used by the official EGCN normalization.
    def scatter_add(src, index, dim=0, dim_size=None):
        if src.ndim != 1 or index.ndim != 1 or dim != 0:
            raise ValueError("GRAPE's native scatter fallback expects 1D dim=0 input.")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        output = src.new_zeros((int(dim_size),))
        return output.scatter_add_(0, index, src)


class EGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, edge_mode, improved=False, cached=False, bias=True, **kwargs):
        super(EGCNConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.edge_mode = edge_mode

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        if edge_mode == 0:
            self.attention_lin = nn.Linear(2 * out_channels + edge_channels, 1)
        elif self.edge_mode == 1:
            self.message_lin = nn.Linear(2 * out_channels + edge_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_attr, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}.".format(self.cached_num_edges, edge_index.size(1))
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_i, x_j, edge_attr, norm):
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i, x_j, edge_attr), dim=-1))
            m_j = attention * x_j
        elif self.edge_mode == 1:
            m_j = torch.cat((x_i, x_j, edge_attr), dim=-1)
            m_j = self.message_lin(m_j)
        return norm.view(-1, 1) * m_j

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.edge_mode == 0:
            aggr_out = aggr_out + x
        return aggr_out
