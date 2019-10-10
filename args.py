# ---- Program configuration ---- #

class Args:
    def __init__(self, data_directory='Graph_datasets/', cuda=False, graph_name='ENZYMES'):
        """
        Class arguments to initialize the VRGC problem parameters
        
        :param data_directory: location of the data (under format as downloaded at https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
        :param cuda: use of CUDA library for GPU-based computation
        :param graph_name: name of the graph dataset
        """

        self.data_directory = data_directory
        self.cuda = cuda
        self.graph_name = graph_name

        self.node_dims = {'MUTAG': 11, 'ENZYMES': 25, 'PROTEINS_full': 80, 'DD': 230, 
             'IMDB-BINARY': 134, 'IMDB-MULTI': 88, 'REDDIT-BINARY': 3068, 'REDDIT-MULTI-5K': 88, 'COLLAB':489}
        self.num_classes = {'MUTAG': 2, 'ENZYMES': 6, 'PROTEINS_full': 2, 'DD': 2, 
             'IMDB-BINARY': 2, 'IMDB-MULTI': 3, 'REDDIT-BINARY': 2, 'REDDIT-MULTI-5K': 5, 'COLLAB':3}

        # dimensions of the neural networks
        self.node_dim = self.node_dims[graph_name]
        self.num_layers = 2
        self.input_size_rnn = self.node_dims[graph_name]  # input size for main RNN
        self.hidden_size_rnn = int(128)
        self.hidden_size_rnn_output = 16
        self.embedding_size_rnn = int(64)
        self.embedding_size_rnn_output = int(8)
        self.embedding_size_output = int(64)

        self.num_class = self.num_classes[graph_name]

        # coefficient of reconstruction loss in the total loss
        self.reco_importance = 0.1

        # ---- Training config ---- #
        self.loss = None
        self.batch_size = 128
        self.epochs = 2000 
        self.epochs_log = 1

        self.lr = 0.001
        self.lr_rate = 0.3
