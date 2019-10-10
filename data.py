import networkx as nx
import numpy as np
import torch
import random
from torch.utils import data
from operator import itemgetter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Some functions of the module data.py are taken from https://github.com/JiaxuanYou/graph-generation


def create_loaders(graphs, args):
    """
    Returns all train and test loaders for the 10-cross validation classification

    :param graphs: list of graphs in networkx format
    :param args: arguments of the problem
    :return: train and test loaders for the 10-cross validation classification
    """
    random.shuffle(graphs)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    labels = np.array([g.graph['label'] for g in graphs])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    skf.get_n_splits(graphs, labels)

    dataloaders_train, dataloaders_test = [], []

    for train_index, test_index in skf.split(graphs, labels):
        graphs_train = itemgetter(*train_index)(graphs)
        graphs_test = itemgetter(*test_index)(graphs)

        dataset_train = GraphSequenceSamplerPytorch(graphs_train, node_dim=args.node_dim)
        dataset_test = GraphSequenceSamplerPytorch(graphs_test, node_dim=args.node_dim)

        dataloaders_train.append(torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size))
        dataloaders_test.append(torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size))

    args.num_class = int(np.max([g.graph['label'] for g in graphs]) - np.min([g.graph['label'] for g in graphs]) + 1)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    args.max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    args.min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # Show graphs statistics

    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(args.max_num_edge, args.min_num_edge))
    print('max previous node: {}'.format(args.node_dim))

    return dataloaders_train, dataloaders_test


def graph_load_batch(data_directory, name):
    """
    Reads graphs from files in a given directory and transforms them into networkx objects

    :param data_directory: data location
    :param name: dataset name (prefix of files to read)
    :return: list of networkx graphs
    """

    print('Loading graph dataset: ' + str(name))
    graph = nx.Graph()

    # load data
    path = data_directory + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)

    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    graph.add_edges_from(data_tuple)

    graph.remove_nodes_from(list(nx.isolates(graph)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []

    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        graph_sub = graph.subgraph(nodes).copy()
        graph_sub.graph['label'] = data_graph_labels[i]

        graphs.append(graph_sub)

    print('Loaded')
    return graphs


def bfs_seq(graph, root):
    """
    Get a BFS transformation of a graph

    :param graph: a networkx graph
    :param root: a node index
    :return: the BFS-ordered node indices
    """

    dictionary = dict(nx.bfs_successors(graph, root))
    to_visit = [root]
    output = [root]
    level_seq = [0]
    level = 1
    while len(to_visit) > 0:
        next_level = []
        while len(to_visit) > 0:
            current = to_visit.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next_level += neighbor
                level_seq += [level] * len(neighbor)
        output += next_level
        to_visit = next_level
        level += 1
    return output


def encode_adj(adjacency, max_prev_node=10):
    """
    Transforms an adjacency matrix to be passed as an input to the RNN

    :param adjacency: adjacency matrix of a graph
    :param max_prev_node: size of the node representation depth (size kept after truncation)
    :return: a sequence of truncated node adjacency
    """

    # pick up lower tri
    adjacency = np.tril(adjacency, k=-1)
    n_nodes = adjacency.shape[0]
    adjacency = adjacency[1:n_nodes, 0:n_nodes - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adjacency.shape[0], max_prev_node))
    for i in range(adjacency.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adjacency[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


class GraphSequenceSamplerPytorch(data.Dataset):
    def __init__(self, graph_list, node_dim=None):
        """

        :param graph_list: list of Networkx graph objects
        :param node_dim: dimensionality of the truncated node dimensionality
        """

        self.adj_all = []
        self.len_all = []
        self.labels = []

        for graph in graph_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(graph)))
            self.len_all.append(graph.number_of_nodes())
            self.labels.append(graph.graph['label'])

        self.labels = [l - np.min(self.labels) for l in self.labels]
        self.max_num_node = max(self.len_all)

        if node_dim is None:
            print('calculating max previous node, total iteration: {}'.format(20000))
            self.node_dim = max(self.calc_max_prev_node(iter=20000))
            print('max previous node: {}'.format(self.node_dim))
        else:
            self.node_dim = node_dim

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        labels_copy = self.labels[idx].copy()
        x_batch = np.zeros((self.max_num_node, self.node_dim))  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros((self.max_num_node, self.node_dim))  # here zeros are padded for small graph

        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        graph = nx.from_numpy_matrix(adj_copy_matrix)

        # ---- Definition of the ordering of the nodes ---- #

        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(graph, start_idx))
        adj_copy_ = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy_.copy(), max_prev_node=self.node_dim)

        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded

        return {'x': x_batch, 'y': y_batch, 'l': labels_copy, 'len': len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()

            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)

            # BFS
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]

            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1 * topk:]
        return max_prev_node


def encode_adj_flexible(adj):
    '''
    Remark: used only if node_dim is not already computed

    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)

    return adj_output
