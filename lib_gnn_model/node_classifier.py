import logging

import torch
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np

from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.sgc.sgc_net_batch import SGCNet
from lib_gnn_model.graphsage.graphsage_net_batch import SAGENet
from lib_gnn_model.gnn_base import GNNBase
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler
import copy
from lib_utils.utils import calc_f1

class NodeClassifier(GNNBase):
    def __init__(self, num_feats, num_classes, args, data=None):
        super(NodeClassifier, self).__init__()

        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.target_model = args['target_model']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {'batch_size': self.args['batch_size'], 'num_workers':0}
        # self.device = 'cpu'
        self.model = self.determine_model(num_feats, num_classes).to(self.device)
        self.data = data

    def determine_model(self, num_feats, num_classes):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        if self.target_model == 'SAGE':
            self.lr, self.decay = 0.01, 0.0
            return SAGENet(num_feats, num_classes)
        elif self.target_model == 'GAT':
            self.lr, self.decay = 0.01, 0.0001
            return GATNet(num_feats, num_classes)
        elif self.target_model == 'GCN':
            self.lr, self.decay = 0.05, 0.0001
            return GCNNet(num_feats, num_classes)
        elif self.target_model == 'GIN':
            self.lr, self.decay = 0.01, 0.001
            return GINNet(num_feats, num_classes)
        elif self.target_model == 'SGC':
            self.lr, self.decay = 0.05, 0.0
            return SGCNet(num_feats, num_classes)
        else:
            raise Exception('unsupported target model')

    def train_model(self, unlearn_info=None):
        # self.logger.info("training model")
        self.model.train()
        self.model.reset_parameters()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        self._gen_train_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        for epoch in range(self.args['num_epochs']):
            self.model.train()
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                if self.args['dataset_name'] == 'ppi':
                    if self.args['inductive'] == 'graphsaint':
                        loss = F.binary_cross_entropy_with_logits(out[batch.train_mask], batch.y[batch.train_mask])
                    else:
                        loss = F.binary_cross_entropy_with_logits(out[:batch.batch_size], batch.y[:batch.batch_size])
                else:
                    if self.args['inductive'] == 'graphsaint':
                        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                    else:
                        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
                
                loss.backward()
                optimizer.step()

            # train_f1, test_f1 = self.evaluate_model()
            # self.logger.info(f'Epoch:{epoch + 1} Train: {train_f1:.4f}, Test: {test_f1:.4f}')
            # print(f"epoch:{epoch+1} training...")
            # print(f'Epoch:{epoch + 1} Train: {train_f1:.4f}, Test: {test_f1:.4f}')

        grad_all, grad1, grad2 = None, None, None
        if self.args["method"] in ["GIF", "IF"]:
            out1 = self.model(self.data.x, self.data.edge_index)
            out2 = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)

            if self.args["unlearn_task"] == "edge":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[2]] = True
                mask2 = mask1
            if self.args["unlearn_task"] == "node":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[0]] = True
                mask1[unlearn_info[2]] = True
                mask2 = np.array([False] * out2.shape[0])
                mask2[unlearn_info[2]] = True
            if self.args["unlearn_task"] == "feature":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[1]] = True
                mask1[unlearn_info[2]] = True
                mask2 = mask1

            loss = F.cross_entropy(out1[self.data.train_mask], self.data.y[self.data.train_mask], reduction='sum')
            loss1 = F.cross_entropy(out1[mask1], self.data.y[mask1], reduction='sum')
            loss2 = F.cross_entropy(out2[mask2], self.data.y[mask2], reduction='sum')
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
            grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
            grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        return (grad_all, grad1, grad2)

    def evaluate_unlearn_F1(self, new_parameters):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1
        self.model.eval()
        out = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)

        y = self.data.y.cpu()
        if self.args['dataset_name'] == 'ppi':
            y_hat = torch.sigmoid(out).cpu().detach().numpy()
            test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
        else:
            y_hat =  out.cpu().detach().numpy()
            test_f1 = calc_f1(y, y_hat, self.data.test_mask)
        return test_f1

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        out = self.model.inference(self.data.x, self.test_loader, self.device)
        y = self.data.y.to(out.device)
        if self.args['dataset_name'] == 'ppi':
            train_f1 = calc_f1(y, out, self.data.train_mask, multilabel=True)
            test_f1 = calc_f1(y, out, self.data.test_mask, multilabel=True)
        else:
            train_f1 = calc_f1(y, out, self.data.train_mask)
            test_f1 = calc_f1(y, out, self.data.test_mask)

        return train_f1, test_f1


    def posterior(self):
        # self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()
        self._gen_test_loader()

        posteriors = self.model.inference(self.data.x, self.test_loader, self.device).to(self.device)

        for _, mask in self.data('test_mask'):
            posteriors = F.softmax(posteriors[mask], dim=-1)

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        logits = self.model.inference(self.data.x, self.test_loader, self.device).to(self.device)
        return logits

    def _gen_train_loader(self):
        temp_data = copy.copy(self.data).cpu()
        
        if self.args['inductive'] == 'cluster-gcn':
            cluster_data = ClusterData(temp_data, num_parts=50, recursive=False)
            self.train_loader = ClusterLoader(cluster_data, batch_size=2048, shuffle=True,
                                         num_workers=0)
        if self.args['inductive'] == 'graphsaint':
            self.train_loader = GraphSAINTRandomWalkSampler(temp_data, batch_size=8000, walk_length=2,
                                                 num_steps=5, sample_coverage=100, num_workers=0)
        else:
            self.train_loader = NeighborLoader(temp_data.contiguous(), input_nodes=temp_data.train_mask,
                                               num_neighbors=[5, 5], shuffle=True, **self.kwargs)

    def _gen_test_loader(self):
        temp_data = copy.copy(self.data).cpu()
        self.test_loader = NeighborLoader(temp_data.contiguous(), input_nodes=None, num_neighbors=[-1], shuffle=False,
                                          **self.kwargs)
        self.test_loader.data.num_nodes = self.data.num_nodes
        self.test_loader.data.n_id = torch.arange(self.data.num_nodes)
        del self.test_loader.data.x, self.test_loader.data.y
