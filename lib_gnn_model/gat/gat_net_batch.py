import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=0.6):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, 8, heads=8, dropout=self.dropout, bias=False))
        # On the Pubmed dataset, use heads=8 in conv2.
        self.convs.append(GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=self.dropout, bias=False))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers - 1):
            x = F.elu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x

    def gnndelete_forward(self, x, edge_index, return_all_emb=False):
        x1 = self.convs[0](x, edge_index)
        x = F.relu(x1)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x2 = self.convs[1](x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        # x_all = F.dropout(x_all, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
