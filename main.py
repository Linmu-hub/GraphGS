import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict
from sklearn import manifold
import matplotlib.pyplot as plt
from torch_geometric.datasets import Amazon

from eval import evaluate
from opt import Opt
from arguments import get_args


def data_spilit(labels, num_cls, dataset):
    np.random.seed(1)
    random.seed(1)

    num_nodes = labels.shape[0]
    rand_indices = np.random.permutation(num_nodes)

    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train_set = list(rand_indices[2000:])

    tr_ratio = []
    count_tr = np.zeros(num_cls)
    if dataset == 'cora':
        count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6])
    elif dataset == 'citeSeer':
        count_tr_ratio = np.array([20, 20, 20, 6, 6, 6])
    elif dataset == 'pubmed':
        count_tr_ratio = np.array([20, 20, 6])
    elif dataset == 'photo':
        count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6, 6])
    elif dataset == 'dblp':
        count_tr_ratio = np.array([20, 20, 6, 6])

    for i in train_set:
        for j in range(num_cls):
            if labels[i] == j:
                count_tr[j] += 1
                break
        if count_tr[labels[i]] <= count_tr_ratio[labels[i]]:
            tr_ratio.append(i)
    train_set = tr_ratio

    test_balanced = []
    count_test = np.zeros(num_cls)
    for i in test:
        for j in range(num_cls):
            if labels[i] == j:
                count_test[j] += 1
                break
        if count_test[labels[i]] <= 100:
            test_balanced.append(i)
    test = test_balanced

    val_bal = []
    count_val = np.zeros(num_cls)
    for i in val:
        for j in range(num_cls):
            if labels[i] == j:
                count_val[j] += 1
                break
        if count_val[labels[i]] <= 30:
            val_bal.append(i)
    val = val_bal

    index = np.arange(0, num_nodes)
    unlable = np.setdiff1d(index, train_set)
    unlable = np.setdiff1d(unlable, val)
    unlable = np.setdiff1d(unlable, test)

    train_y = []
    for i in train_set:
        train_y.append(int(labels[i]))

    val_y = []
    for i in val:
        val_y.append(int(labels[i]))
    test_y = []
    for i in test:
        test_y.append(int(labels[i]))

    return train_set, train_y, val, val_y, test, test_y, unlable


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    nodes = []
    node_map = {}   # 节点：索引
    label_map = {}  # 标签：数字表示
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            nodes.append(i)
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, nodes


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_cls = 7
    feat_data, labels, adj_lists, nodes = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls, 'cora')

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)

    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls, torch.Tensor(nodes)


def run_citeSeer():
    np.random.seed(1)
    random.seed(1)

    file_path = 'citeSeer/CiteSeer.pt'
    feat_data, labels, adj_lists, nodes = load_citeSeer(file_path)
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()
    num_cls = np.max(labels) + 1
    print(num_cls)
    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls, 'citeSeer')

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)

    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls, torch.Tensor(nodes)


def load_citeSeer(path):
    data = torch.load(path)
    x = data[0]['x']
    edge_index = data[0]['edge_index']
    y = data[0]['y']

    x = np.array(x)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    edge_index = np.array(edge_index)

    nodes = [i for i in range(y.shape[0])]

    graph = defaultdict(set)

    for i, node in enumerate(nodes):
        graph[i].add(i)

    edge_index = np.array(edge_index)
    start_nodes = edge_index[0]
    connected_nodes = edge_index[1]

    for start, end in zip(start_nodes, connected_nodes):
        graph[start].add(end)
        graph[end].add(start)

    print(x.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)

    result = dict(zip(unique, counts))
    print(result)
    return x, y, graph, list(range(len(x)))


def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    if args.dataset == 'cora':
        train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls, nodes = run_cora()
    elif args.dataset == 'citeSeer':
        train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels, num_cls, nodes = run_citeSeer()

    opt = Opt(train_x, train_y, unlable, val_x, val_y, features, adj_lists, test_x, test_y, num_cls, args, labels, nodes)

    evaluate(opt=opt, test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists,
                labels=labels, dataset=args.dataset)


if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    main(args)


