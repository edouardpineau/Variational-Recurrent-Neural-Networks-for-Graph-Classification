from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RecurrentEmbedding(nn.Module):
    """
    Recurrent embedding function: end of the blue block in the paper
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, is_cuda=False):
        super(RecurrentEmbedding, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda
        self.input = nn.Linear(input_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.relu = nn.ReLU()

        # initialize

        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data =nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.is_cuda:
            self.hidden = self.hidden.cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        input = self.input(input_raw)
        input = self.relu(input)
        
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        
        return output_raw


class VAR(nn.Module):
    """
    Variational regularization: green block in the paper
    """

    def __init__(self, h_size, embedding_size, y_size, is_cuda=False):
        super(VAR, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        self.is_cuda = is_cuda

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data =nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        
        if self.training:
            eps = torch.randn(z_sgm.size())
            if self.is_cuda:
                eps = eps.cuda()
                
            z = eps*z_sgm + z_mu
        else:
            z = z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms


class RecurrentClassifier(nn.Module):
    """
    Recurrent classification: yellow block in the paper
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, num_class=None, is_cuda=False):
        super(RecurrentClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, num_class),
                nn.Softmax(dim=1))

        self.relu = nn.ReLU()
        self.hidden = None

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data =nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.is_cuda:
            self.hidden = self.hidden.cuda()

    def forward(self, input_raw, pack=True, input_len=None):
        sample = input_raw
        if pack:
            sample = pack_padded_sequence(sample, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(sample, self.hidden)
        l_pred = self.output(self.hidden[-1])
        return l_pred, output_raw
