# Variational Recurrent Neural Networks for Graph Classification

This work was presented at the ICLR 2019 workshop session [Representation Learning on Graphs and Manifolds](https://rlgm.github.io/). 

Our paper can be found here: https://rlgm.github.io/papers/9.pdf

The notebook VRGC.ipynb requires the data that can be downloaded [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

### Abstract

We address the problem of graph classification based only on structural information. Inspired by natural language processing techniques (NLP), our model sequentially embeds information to estimate class membership probabilities. Besides, we experiment with NLP-like variational regularization techniques, making the model predict the next node in the sequence as it reads it. We experimentally show that our model achieves state-of-the-art classification results on several standard molecular datasets. Finally, we perform a qualitative analysis and give some insights on whether the node prediction helps the model better classify graphs.

### Multi-task learning

Multi-task learning is a powerful leverage to learn rich representation in NLP [1]. We propose to use it for our problem.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_macro.png" width="700"></p>

#### Graph preprocessing

We use a BFS node ordering procedure to transform graph into sequence of nodes as in [2]. 

##### Breadth-first search with random root R for graph enumeration

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/BFS.png" width="400"></p>
<p align="center">Figure 2: Example of a BFS node ordering.</p>

##### Sequencial truncated node adjacency

Each node is only related to its two closest neighbors in the order of the BFS to get a low dimensional sequence of nodes. 

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/SeqAdj.png" width="500"></p>
<p align="center">Figure 3: Example of a BFS node ordering.</p>

##### Complete graph-to-sequence embedding

Each node embedding contains both current and previous node BFS-related adjacency information thanks to RNN memory structure.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_0.png" width="600"></p>
<p align="center">Figure 4: Example of a BFS node ordering.</p>

#### Recurrent neural network for sequence classification

The fully connected (FC) classifier is fed with sequence of the truncated BFS-ordered embedded node sequence.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_2.png" width="600"></p>
<p align="center">Figure 5: Recurrent classifier for for sequence classification.</p>

#### Variational autoregressive (VAR) node prediction

A node prediction task is added to help the classifier. The task is performed by a variational autoencoder feed with the same sequence of embedded nodes than the recurrent classifier.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/archi_1.png" width="600"></p>
<p align="center">Figure 6: Variational autoregressive node prediction.</p>


### Results

- VRGC is not structurally invariant to node indexing, like in our previous [graph classification work](https://github.com/edouardpineau/A-simple-baseline-algorithm-for-graph-classification). However our model learns node indexing invariance from numerous training iterations on randomly-rooted BFS-ordered sequential graph embedding.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/Projection.png" width="400"></p>
<p align="center">Figure 7: TSNE projection of the latent state preceding classification for five graphs from four distinct classes, each initiated with 20 different BFS. Colors and markers represent the respective classes of the graphs.</p>

&nbsp;

- VAR helps the model finding a more meaningful latent representation for classification while graph dataset becomes larger, with marginal extra computational cost with respect to RNNs.

<p align="center"><img src="https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification/raw/master/images/table.png" width="500"></p>
<p align="center">Figure 8: Classification results.</p>

### Citing

    @article{pineau2019variational,
              title={Variational recurrent neural networks for graph classification},
              author={Pineau, Edouard and de Lara, Nathan},
              journal={Representation Learning on Graphs and Manifolds Workshop (ICLR)},
              year={2019}
}


### Datasets and references

All datasets can be found here: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

[1] V. Sanh, T. Wolf, and S. Ruder. A hierarchical multi-task approach for learning embeddings from semantic tasks. arXiv preprint arXiv:1811.06031, 2018.

[2] GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models, Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, Jure Leskovec ; Proceedings of the 35th International Conference on Machine Learning, PMLR 80:5708-5717, 2018.
