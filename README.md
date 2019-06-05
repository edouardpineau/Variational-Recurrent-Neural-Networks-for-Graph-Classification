# Variational-Recurrent-Neural-Networks-for-Graph-Classification
Github page for the paper "Variational Recurrent Neural Networks for Graph Classification" presented at the RLGM workshop of ICLR 2019

## Abstract

We address the problem of graph classification based only on structural information. Inspired by natural language processing techniques (NLP), our model sequentially embeds information to estimate class membership probabilities. Besides, we experiment with NLP-like variational regularization techniques, making the model predict the next node in the sequence as it reads it. We experimentally show that our model achieves state-of-the-art classification results on several standard molecular datasets. Finally, we perform a qualitative analysis and give some insights on whether the node prediction helps the model better classify graphs.

## Multi-task learning

Multi-task learning is a powerful leverage to learn rich representation in NLP [1]. We propose to use it for our problem.

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_macro.png" width="800">
Figure 1: Schematic macro view of our model. 

### Graph preprocessing

We use a BFS node ordering procedure to transform graph into sequence of nodes as in [2]. 

##### Breadth-first search with random root $R$ for graph enumeration

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/BFS.png" width="800">
Figure 2: Example of a BFS node ordering.

##### Sequencial truncated node adjacency

Each node is only related to its two closest neighbors in the order of the BFS to get a low dimensional sequence of nodes. 

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/SeqAdj.png" width="800">
Figure 3: Example of a BFS node ordering.

##### Complete graph-to-sequence embedding

Each node embedding contains both current and previous node BFS-related adjacency information thanks to RNN memory structure.

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_0.png" width="800">
Figure 4: Example of a BFS node ordering.

### Recurrent neural network for sequence classification

The fully connected (FC) classifier is fed with sequence of the truncated BFS-ordered embedded node sequence.

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_2.png" width="800">
Figure 5: Recurrent classifier for for sequence classification. 

### Variational autoregressive (VAR) node prediction

A node prediction task is added to help the classifier. The task is performed by a variational autoencoder feed with the same sequence of embedded nodes than the recurrent classifier.

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_1.png" width="800">
Figure 6: Variational autoregressive node prediction.


## Contribution

- VRGC is not structurally invariant to node indexing, it learns it from numerous training iterations on randomly-rooted BFS-ordered sequential graph embedding

- VAR helps the model finding a more meaningful latent representation for classification while graph dataset becomes larger, with marginal extra computational cost with respect to RNNs

<img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/table.png" width="800">
Figure 6: Variational autoregressive node prediction.