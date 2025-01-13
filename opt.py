import math

import numpy as np
import random
import torch
from sklearn import metrics
from collections import deque

from sklearn.metrics import classification_report

from model import MeanAggregator, SupervisedGraphSage, Encoder_bl
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Opt:
    def __init__(self, node_label, node_label_y, node_unlabel, val_x, val_y, features, adj_lists, test_x, test_y,
                 num_cls, args, all_label, nodes):
        self.args = args
        self.all_nodes = nodes
        self.features = features
        self.all_labels = all_label
        self.val_x = val_x
        self.val_y = val_y
        self.train_node_ori = node_label.clone().detach()
        self.train_node_y_ori = node_label_y.clone().detach()
        self.test_x = test_x
        self.test_y = test_y
        self.num_cls = num_cls
        self.adj_lists = adj_lists

        self.classifier = self.cls_model(features, adj_lists, features.weight.shape[1], 128, 'weight')

        self.unlabel_set = node_unlabel
        self.candidate_node = []
        self.candidate_node_y = []
        self.pred_can = []
        self.emb_can = []
        self.pred_v = []

        self.count_tr_ratio, self.count_class, self.class_less = self.ratio_less()
        print(self.class_less, self.count_tr_ratio)
        self.count = 0
        self.done = 0
        self.emb_sum = torch.zeros((128, num_cls))
        self.emb_cen = torch.zeros((128, num_cls))
        self.mean = 0
        self.f1 = 0
        self.f1_max = 0

    def ratio_less(self):
        split_loc = math.ceil(self.num_cls / 2)
        print(split_loc)
        count_tr_ratio = []
        class_less = []
        for i in range(self.num_cls):
            if i < split_loc:
                count_tr_ratio.append(self.args.train_node_num)
            else:
                count_tr_ratio.append(int(self.args.train_node_num * self.args.ratio))
        # count_tr_ratio = np.array(count_tr_ratio)
        for k in range(split_loc, self.num_cls):
            class_less.append(k)
        return count_tr_ratio, count_tr_ratio, class_less

    def reset(self):
        self.count = 0
        self.done = 0
        self.train_node = self.train_node_ori
        self.train_node_y = self.train_node_y_ori
        self.supplement_emb = torch.zeros(128)

        self.calculate_connect(self.train_node)

        pre_train = self.cls_train(300, self.classifier, self.train_node, self.train_node_y)

        pre_train.eval()
        pred_l, emb_l = pre_train.forward(self.train_node)

        emb_l = emb_l.cpu()

        self.emb_l = torch.sum(emb_l.detach(), 1)  # [128, 98] 按行求和 -> 128
        print("--"*30)

        pred_u, emb_u = pre_train.forward(self.unlabel_set)

        pred_u = pred_u.cpu()
        emb_u = emb_u.cpu()

        self.candidate_node, self.candidate_node_y = self.SelectNode(emb_l, pred_u, emb_u)

    def get_nei_node(self, k):
        nei = set([])
        for i, node in enumerate(self.train_node):
            if self.train_node_y[i] == k:
                nei = nei.union(set(self.adj_lists.get(int(node))))

        return nei

    def dsw(self):
        pre_train = self.cls_train(30, self.classifier, self.train_node, self.train_node_y)

        pre_train.eval()
        pred_l, emb_l = pre_train.forward(self.train_node)

        emb_l = emb_l.cpu()

        self.emb_l = torch.sum(emb_l.detach(), 1)  # [128, 98] 按行求和 -> 128

        pred_u, emb_u = pre_train.forward(self.unlabel_set)

        pred_u = pred_u.cpu()
        emb_u = emb_u.cpu()

        self.candidate_node, self.candidate_node_y = self.SelectNode(emb_l, pred_u, emb_u)

    def plot_embeddings(self, embeddings, data, dataset):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score

        Y = data
        # Y = get_onehot(Y)

        # Y = Y.numpy()
        embeddings = embeddings

        # emb_list = []
        # for k in range(Y.shape[0]):
        #     emb_list.append(embeddings[k])

        emb_list = embeddings

        model = TSNE(n_components=2)
        node_pos = model.fit_transform(emb_list)

        color_idx = {}
        for i in range(Y.shape[0]):
            # label = Y[i].item()  # 将张量转换为整数
            color_idx.setdefault(Y[i], [])
            # color_idx.setdefault(label, [])
            color_idx[Y[i]].append(i)
            # color_idx[label].append(i)
        plt.figure()
        # ax = Axes3D(fig)
        # colors = {0: 'red', 1: 'green', 2: 'blue'}

        for c, idx in color_idx.items():
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=10, alpha=0.7)

        sil_score = silhouette_score(node_pos, Y)
        plt.title(f'Silhouette Score: {sil_score:.2f}')
        plt.savefig(f"./tse_graph/SAGE-M/{dataset}_{0}.svg", format='svg')
        plt.show()

    def step(self, dataset):

        retrain = self.cls_train(10, self.classifier, self.train_node, self.train_node_y)

        self.pred_v, _ = retrain.forward(self.val_x)

        report = classification_report(self.val_y.cpu(), self.pred_v.cpu().detach().numpy().argmax(axis=1), digits=4,
                                       output_dict=True)
        # 从生成的字典中提取每个类的准确率
        accuracies = {str(key): value['precision'] for key, value in report.items() if key.isdigit()}
        sorted_items = sorted(accuracies.items(), key=lambda x: x[1])[:1]

        # 从排序后的元组中提取键
        keys = [int(item[0]) for item in sorted_items]
        # print(keys)
        # print(len(self.candidate_node), len(self.candidate_node_y))

        # all_nodes = self.all_nodes.long()
        # retrain.eval()
        # _, h = retrain.forward(all_nodes)
        # out_h = h.detach().cpu().numpy()
        # all_labels = self.all_labels.reshape(self.all_labels.shape[0])
        # self.plot_embeddings(out_h.T, all_labels, "cora")

        if dataset == 'cora':
            # 需要加入的数量
            add_node_num_flag = [0, 0, 0, 0, 20, 20, 20]
            add_node_num = [0, 0, 0, 0, 0, 0, 0]
        elif dataset == 'citeSeer':
            add_node_num_flag = [0, 0, 0, 20, 20, 20]
            add_node_num = [0, 0, 0, 0, 0, 0]

        self.train_node = list(self.train_node)
        self.train_node_y = list(self.train_node_y)

        bp = True
        while bp:
            for k in range(self.num_cls):
                k = int(k)
                nei_nodes = self.get_nei_node(k)

                for i, y in enumerate(self.candidate_node_y):  # 添加节点
                    if add_node_num[k] >= add_node_num_flag[k]:
                        break
                    if k == y and self.candidate_node[i] in nei_nodes:
                        # print("1", y, self.all_labels[self.candidate_node[i]])
                        self.train_node.append(self.candidate_node[i])
                        self.train_node_y.append(y)
                        self.unlabel_set = self.unlabel_set[self.unlabel_set != self.candidate_node[i]]
                        add_node_num[k] += 1
                        self.count_class[k] += 1
                m = 0
                for i, y in enumerate(self.candidate_node_y):  # 添加节点
                    if add_node_num[k] >= add_node_num_flag[k]:
                        break
                    if k == y and m < 20 and self.candidate_node[i] not in nei_nodes:  # [10-20]
                        # print("2", y, self.all_labels[self.candidate_node[i]])
                        self.train_node.append(self.candidate_node[i])
                        self.train_node_y.append(y)
                        self.unlabel_set = self.unlabel_set[self.unlabel_set != self.candidate_node[i]]
                        add_node_num[k] += 1
                        self.count_class[k] += 1
                        m += 1
            bp = not all(add_node_num[i] >= add_node_num_flag[i] for i in range(len(add_node_num)))
            self.dsw()

        del self.classifier
        torch.cuda.empty_cache()
        print('_'* 30)

    def SelectNode(self, emb_l, pred_u, emb_u):  # choose unlabel nodes based on distance
        emb_sum = torch.zeros((128, self.num_cls))

        for i in range(len(self.train_node)):
            emb_sum[:, self.train_node_y[i]] = emb_sum[:, self.train_node_y[i]] + emb_l[:, i]

        nums = [0, 0, 0, 0, 0, 0, 0, 0]
        max_indices = torch.argmax(pred_u, dim=1)
        for i in range(len(pred_u)):
            emb_sum[:, max_indices[i]] = emb_sum[:, max_indices[i]] + emb_u[:, i]
            nums[max_indices[i]] += 1

        emd_cen = torch.zeros((128, self.num_cls))
        for i in range(emb_sum.shape[1]):
            emd_cen[:, i] = emb_sum[:, i] / (self.count_class[i] + nums[i])

        dict_node = defaultdict(list)
        dict_unemb = defaultdict(list)

        pred_y = pred_u.data.numpy().argmax(axis=1)
        unique, counts = np.unique(pred_y, return_counts=True)
        result = dict(zip(unique, counts))
        print('pre_y', result)

        for i in range(len(self.unlabel_set)):
            dict_node[pred_y[i]].append(self.unlabel_set[i])
            dict_unemb[pred_y[i]].append(emb_u[:, i].detach().numpy())
        node_num = 50  # [30 | 50]

        c = 0
        supplement = np.zeros(len(self.class_less) * node_num, dtype=int)
        supplement_y = np.zeros(len(self.class_less) * node_num, dtype=int)

        for i in self.class_less:
            cen = emd_cen[:, i].detach().numpy()  # shape=128

            node = np.array(dict_node.get(i))

            emb = np.array(dict_unemb.get(i))

            dis = []

            selnodes = node[0:node_num]

            for j in range(len(node)):
                distance = np.linalg.norm(cen - emb[j])
                if j < node_num:
                    dis.append(distance)
                else:
                    dis_max = max(dis)
                    idx_max = dis.index(dis_max)
                    if distance < dis_max:
                        dis[idx_max] = distance
                        selnodes[idx_max] = node[j]

            dis_node = zip(dis, selnodes)
            dis_node_sort = sorted(dis_node, key=lambda x: x[0])
            dis_sort, selnodes_sort = [list(x) for x in zip(*dis_node_sort)]
            p = 0
            for x in range(len(selnodes)):
                supplement[p + c] = selnodes_sort[x]
                supplement_y[p + c] = i
                p += len(self.class_less)
            c += 1

        return supplement, supplement_y

    def calculate_connect(self, nodes):  # 计算节点间的连接数
        nodes = np.array(nodes)
        nodes = set(nodes)
        edges_num = 0
        cross_edges_num = 0
        for node in nodes:
            edges = self.adj_lists[int(node)]
            edges_num += len(edges)
            cross_edges_num += len((set(edges) & nodes))
        print('nodes num', len(nodes), 'cross_edges_num', cross_edges_num, 'edges_num', edges_num)

    def cls_model(self, features, adj_lists, fea_size, hidden, loss_fun):
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder_bl(features, fea_size, hidden, adj_lists, agg1, gcn=True, cuda=True)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
        enc2 = Encoder_bl(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=True)
        enc1.num_samples = None
        enc2.num_samples = None
        graphsage = SupervisedGraphSage(self.num_cls, enc2, loss_fun)

        return graphsage

    def cls_train(self, epoch, graphsage, train_x, train_y):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7, weight_decay=1e-3)
        graphsage.train()
        for batch in range(epoch):
            batch_nodes = train_x[:256]
            batch_y = train_y[:256]

            c = list(zip(train_x, train_y))
            random.shuffle(c)
            train_x, train_y = zip(*c)

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, batch_y)
            # print('loss2', loss)
            loss.backward()
            optimizer.step()

        return graphsage
