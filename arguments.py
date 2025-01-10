import argparse


def get_args():
    """
        Description:
        Parses arguments at command line.

        Parameters:
            None

        Return:
            args - the arguments parsed
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_node_num', dest='train_node_num', type=int, default=20)  # number of nodes used for training
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.3)  # imbalance ratio
    parser.add_argument('--isGCN', dest='isGCN', type=bool, default=True)  # can be use GCN

    parser.add_argument('--dataset', dest='dataset', type=str, default='cora')

    args = parser.parse_args()

    return args