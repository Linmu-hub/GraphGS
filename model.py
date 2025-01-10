import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.autograd import Variable
import random
import numpy as np
from arguments import get_args
args = get_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))

        return combined


"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        zero_neigh_mask = (num_neigh == 0).squeeze()
        mask[zero_neigh_mask] = 0
        num_neigh[zero_neigh_mask] = 1

        mask = mask.div(num_neigh)
        unique_nodes_list = [int(node) for node in unique_nodes_list]
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)

        return to_feats


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, loss_fun):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc.to(device)
        if loss_fun == 'weight':
            class_weights = []
            for i in range(num_classes):
                if i < math.ceil(num_classes/2):
                    class_weights.append(1)
                else:
                    class_weights.append(4.)
            class_weights = torch.tensor(class_weights, device=device)
            self.xent = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim)).to(device)
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t(), embeds

    def loss(self, nodes, labels):
        if isinstance(nodes, np.int64):
            nodes = torch.tensor([nodes], dtype=torch.long)
        labels = torch.LongTensor(labels).to(device)
        scores, embeds = self.forward(nodes)
        return self.xent(scores, labels)


def cls_model(features, adj_lists, fea_size, hidden, num_cls, loss_fun):
    isGCN= args.isGCN
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, fea_size, hidden, adj_lists, agg1, gcn=isGCN, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden, adj_lists, agg2,
                       base_model=enc1, gcn=isGCN, cuda=True)
    enc1.num_samples = 5
    enc2.num_samples = 5
    graphsage = SupervisedGraphSage(num_cls, enc2, loss_fun)

    return graphsage


def cls_train(graphsage, train_x, train_y):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    for batch in range(100):
        batch_nodes = train_x[:256]
        batch_y = train_y[:256]

        c = list(zip(train_x, train_y))
        random.shuffle(c)
        train_x, train_y = zip(*c)

        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, batch_y)
        # print("loss", loss)
        loss.backward()
        optimizer.step()

    return graphsage


class Encoder_bl(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder_bl, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim))
        self.weight_2 = nn.Parameter(
            torch.FloatTensor(self.feat_dim, 1024))
        self.z = nn.Parameter(
            torch.FloatTensor(1024, 1))

        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.weight_2, gain=1.414)
        init.xavier_uniform_(self.z)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                self.num_sample)

        if isinstance(nodes, np.ndarray):
            nodes = torch.from_numpy(nodes).long()
        if isinstance(nodes, (list, tuple)):
            nodes = torch.tensor(nodes, dtype=torch.long)

        nodes = nodes.to(device)
        feature_nodes = self.features(nodes)

        s_nodes = torch.mm(feature_nodes, self.weight_2)
        s_neighs = torch.mm(neigh_feats, self.weight_2)

        s_nodes = torch.tanh(s_nodes)
        s_neighs = torch.tanh(s_neighs)

        u_nodes = torch.mm(s_nodes, self.z)
        u_neighs = torch.mm(s_neighs, self.z)

        sum_nodes = torch.sum(u_nodes) / s_nodes.shape[0]
        sum_neighs = torch.sum(u_neighs) / s_neighs.shape[0]

        values = torch.stack((sum_nodes, sum_neighs))
        max_value = values.max()
        att2 = F.softmax(values - max_value, dim=0)
        combined = F.relu(att2[0] * feature_nodes + att2[1] * neigh_feats)

        combined = F.relu(self.weight.mm(combined.t()))
        combined = F.dropout(combined, 0.2, training=self.training)

        return combined







