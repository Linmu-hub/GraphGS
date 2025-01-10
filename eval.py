import time
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, silhouette_score
from model import cls_model, cls_train
import matplotlib.pyplot as plt
from arguments import get_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_embeddings(embeddings, data, dataset):
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
    plt.savefig(f"./tse_graph/SAGE/{dataset}_{0}.svg", format='svg')
    plt.show()


def evaluate(opt, test_x, test_y, features, adj_lists, labels, dataset):

    start_time = time.time()

    opt.train_node = opt.train_node_ori
    opt.train_node_y = opt.train_node_y_ori

    opt.reset()
    opt.step(dataset)

    train_x = opt.train_node
    train_y = opt.train_node_y

    num_cls = np.max(labels) + 1
    count_sup = np.zeros(num_cls)
    for i in range(0, len(train_x)):
        count_sup[train_y[i]] += 1
    print('count_sup', count_sup)

    opt.calculate_connect(train_x)

    classifer = cls_model(features, adj_lists, features.weight.shape[1], 128, num_cls, 'ave')
    final_train = cls_train(classifer, train_x, train_y)

    times = time.time() - start_time
    print(f"类平衡时间: {times:.2f} 秒")

    start_time = time.time()

    final_train.eval()
    pred_test, h = final_train.forward(test_x)
    pred_test = pred_test.cpu()

    times = time.time() - start_time
    print(f"预测时间: {times*1000:.2f} 毫秒")

    one_hot = np.identity(num_cls)[test_y]

    report = classification_report(test_y.cpu(), pred_test.data.numpy().argmax(axis=1), digits=4, output_dict=True)
    print(classification_report(test_y.cpu(), pred_test.data.numpy().argmax(axis=1), digits=4))
    # 从生成的字典中提取每个类的准确率
    accuracies = {str(key): value['precision'] for key, value in report.items() if key.isdigit()}
    print(accuracies)
    auc = roc_auc_score(one_hot, pred_test.data.numpy(), average='macro')
    print("Test roc_auc:", auc)

    # _, h = final_train.forward(env.all_nodes)
    # out_h = h.detach().cpu().numpy()
    # # np.savez(Path.cwd().joinpath("out_emd"), out_h)
    #
    # all_labels = env.all_labels.reshape(env.all_labels.shape[0])
    #
    # plot_embeddings(out_h.T, all_labels, "cora")


