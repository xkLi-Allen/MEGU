import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, 256))
        self.convs.append(SAGEConv(256, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x

    def gnndelete_forward(self, x, edge_index, return_all_emb=False):
        x1 = self.convs[0](x, edge_index)
        x = F.relu(x1)
        # x = F.dropout(x, p=0.5, training=self.training)
        x2 = self.convs[1](x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
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
