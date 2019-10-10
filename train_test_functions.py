import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as functional

from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# REMARK: the functions 'pad_packed_sequence' and 'pack_padded_sequence' enable to deal
# with sequences of different length in recurrent neural networks


def train_vrgc_epoch(epoch, args, rnn_embedding, var, rnn_classifier, dataloader_train,
                     optimizer_rnn, optimizer_var, optimizer_classifier):
    """
    Training procedure for the VRGC model (rnn+var+classifier)

    :param epoch: number of training epochs
    :param args: arguments of the problem
    :param rnn_embedding: recurrent embedding
    :param var: variational regularizer
    :param rnn_classifier: recurrent classifier
    :param dataloader_train: train set loader
    :param optimizer_rnn: recurrent embedding Adam optimizer
    :param optimizer_var: variational regularizer Adam optimizer
    :param optimizer_classifier: recurrent classifier Adam optimizer
    """

    rnn_embedding.train()
    var.train()
    rnn_classifier.train()

    loss_sum, loss_c_sum, accuracy = 0, 0, 0
    tot_data = 0

    for batch_idx, data in enumerate(dataloader_train):
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()

        label_unsorted = data['l'].long()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)

        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        # Sort input

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()

        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        labels = torch.index_select(label_unsorted, 0, sort_index)

        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            labels = labels.cuda()

        rnn_embedding.zero_grad()
        var.zero_grad()
        rnn_classifier.zero_grad()

        rnn_embedding.init_hidden(batch_size=x_unsorted.size(0))
        rnn_classifier.init_hidden(batch_size=x_unsorted.size(0))

        h = rnn_embedding(x, pack=True, input_len=y_len)
        y_pred, z_mu, z_logvar = var(h)

        y_pred = torch.sigmoid(y_pred)
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_logvar = pack_padded_sequence(z_logvar, y_len, batch_first=True)
        z_logvar = pad_packed_sequence(z_logvar, batch_first=True)[0]

        rnn_classifier.init_hidden(x.size(0))
        pred_labels, var_raw = rnn_classifier(h, pack=True, input_len=y_len)
        loss_classifier = functional.cross_entropy(pred_labels, labels)

        loss_bce = args.loss(y_pred, y)

        loss_kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        loss_kl /= y.size(0) * y.size(1) * sum(y_len)

        loss = args.reco_importance * (loss_bce + loss_kl) + loss_classifier

        loss.backward()

        optimizer_var.step()
        optimizer_rnn.step()
        optimizer_classifier.step()

        accuracy += loss_classifier.item()
        tot_data += x_unsorted.size(0)

    if epoch % args.epochs_log == 0:
        print('Dataset: {}, Epoch: {}/{}, train bce loss: {:.3f}, train kl loss: {:.3f}, classifier loss: {:.3f}'
              .format(args.graph_name, epoch, args.epochs, loss_bce.item(), loss_kl.item(), accuracy))


def test_vrgc_epoch(args, rnn_embedding, var, rnn_classifier, dataloader_test):
    """
    Accuracy evaluation at test time

    :param args: arguments of the problem
    :param rnn_embedding: recurrent embedding
    :param var: variational regularization
    :param rnn_classifier: recurrent classifier
    :param dataloader_test: test set loader
    :return: test accuracy
    """

    rnn_embedding.eval()
    var.eval()
    rnn_classifier.eval()

    loss_sum, accuracy, tot_data = 0, 0, 0
    total_predicted_labels, total_labels = [], []

    for batch_idx, data in enumerate(dataloader_test):

        x_unsorted = data['x'].float()
        label_unsorted = data['l'].long()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, :y_len_max, :]

        # Initialize gradients and LSTM hidden state according to batch size

        rnn_embedding.init_hidden(batch_size=x_unsorted.size(0))
        rnn_classifier.init_hidden(batch_size=x_unsorted.size(0))

        # Sort input

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        labels = torch.index_select(label_unsorted, 0, sort_index)

        if args.cuda:
            x = x.cuda()
            labels = labels.cuda()

        h = rnn_embedding(x, pack=True, input_len=y_len)

        # Standard GRU classification

        rnn_classifier.init_hidden(x.size(0))
        pred_labels, var_raw = rnn_classifier(h, pack=True, input_len=y_len)

        accuracy += torch.sum((labels == pred_labels.topk(1)[-1].squeeze()).float()).item()
        tot_data += x_unsorted.size(0)

        total_predicted_labels.append(pred_labels)
        total_labels.append(labels)
        
    return accuracy / tot_data, total_predicted_labels, total_labels


def vote_test(args, rnn_embedding, var, rnn_classifier, dataloader_test, num_iteration=10):
    """
    Aggregation of the results at test time

    :param args: arguments of the problem
    :param rnn_embedding: recurrent embedding
    :param var: variational regularization
    :param rnn_classifier: recurrent classifier
    :param dataloader_test: test set loader
    :param num_iteration: number N of times a graph is tested (with random BFS root)
    :return: test accuracy
    """

    scores = []
    acc, pred_labels, true_labels = test_vrgc_epoch(args, rnn_embedding, var, rnn_classifier,
                                                       dataloader_test)

    vote = torch.cat(pred_labels, dim=0).cpu().data.numpy()

    for _ in np.arange(num_iteration):
        acc, pred_labels, true_labels = test_vrgc_epoch(args, rnn_embedding, var, rnn_classifier, dataloader_test)
        vote = np.maximum(vote, torch.cat(pred_labels, dim=0).cpu().data.numpy())

        scores.append(acc)

    predicted_labels = np.argmax(vote, axis=1)
    accuracy_vote = np.sum((torch.cat(true_labels, dim=0).cpu().data.numpy() == predicted_labels)) / predicted_labels.shape[0]

    if args.cuda:
        del pred_labels
        torch.cuda.empty_cache()

    return accuracy_vote, scores, predicted_labels, torch.cat(true_labels), vote


def classifier_train(args, dataloader_train, dataloader_test, rnn_embedding, var, rnn_classifier):
    """

    :param args: arguments of the problem
    :param dataloader_train: train set loader (90% of the data)
    :param dataloader_test: test set loader (10% of the data)
    :param rnn_embedding: recurrent embedding
    :param var: variational regularizer
    :param rnn_classifier: recurrent classifier
    :return: all the test accuracies for the 10 folds cross-validation
    """

    epoch = 1

    # Initialize optimizers

    optimizer_rnn = optim.Adam(rnn_embedding.parameters(), lr=args.lr)
    optimizer_var = optim.Adam(var.parameters(), lr=args.lr)
    optimizer_classifier = optim.Adam(rnn_classifier.parameters(), lr=args.lr)

    # Start main loop

    all_test_losses = []

    while epoch <= args.epochs:
        train_vrgc_epoch(epoch, args, rnn_embedding, var, rnn_classifier,
                         dataloader_train, optimizer_rnn, optimizer_var, optimizer_classifier)

        if epoch % 50 == 0:
            # For the published Github version, we screen the test accuracy every 50 epochs

            accuracy_test, scores, predicted_labels, true_labels, vote = vote_test(args,
                                                                                   rnn_embedding,
                                                                                   var,
                                                                                   rnn_classifier,
                                                                                   dataloader_test,
                                                                                   num_iteration=10)
            all_test_losses.append(accuracy_test)

            print('Epoch: {}, Test accuracy: {:.3f}'.format(epoch, accuracy_test))

        epoch += 1

    return all_test_losses
