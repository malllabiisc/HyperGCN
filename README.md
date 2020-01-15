# HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs

[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://iclr.cc/) [![Paper](http://img.shields.io/badge/paper-arxiv.1809.02589-B31B1B.svg)](https://arxiv.org/abs/1809.02589) 

Source code for [NeurIPS 2019](https://iclr.cc/) paper: [**HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs**](https://papers.nips.cc/paper/8430-hypergcn-a-new-method-for-training-graph-convolutional-networks-on-hypergraphs)

![](./hmlap.png)

**Overview of HyperGCN:** *Given a hypergraph and vertex features, HyperGCN approximates the Hypergraph by a graph in which each hyperedge is approximated by a subgraph consiting of an edge between maximally disparate vertices and edges between each and every other vertex (mediator) of the hyperedge.*

### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.

### Dataset:

- DBLP dataset used in the paper is included in the `data` directory.
- Other datasets (Cora, Citeseer, Pubmed) used in the paper can be downloaded from [here](https://linqs.soe.ucsc.edu/data)

### Training model (Node classifiction):

- To start training run:

  ```shell
  python hypergcn.py -mediators True -split 0
  ```

  - `-mediators` denotes whether to use mediators (True) or not (False) 
  - `-split` is the train-test split number
  

### Citation:

Please cite the following paper if you use this code in your work.

```bibtex
@incollection{hypergcn_neurips19,
title = {HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs},
author = {Yadati, Naganand and Nimishakavi, Madhav and Yadav, Prateek and Nitin, Vikram and Louis, Anand and Talukdar, Partha},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 32},
pages = {1509--1520},
year = {2019},
publisher = {Curran Associates, Inc.}
}

```
