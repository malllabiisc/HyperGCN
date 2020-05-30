import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load(args):
    """
    parses the dataset
    """
    dataset = parser(args.data, args.dataset).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, args.dataset, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    return dataset, train, test



class parser(object):
    """
    an object for parsing data
    """
    
    def __init__(self, data, dataset):
        """
        initialises the data directory 

        arguments:
        data: coauthorship/cocitation
        dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """
        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, dataset)
        self.data, self.dataset = data, dataset

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """
        
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)
            print("number of hyperedges is", len(hypergraph))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """
        
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)