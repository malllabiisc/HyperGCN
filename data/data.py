import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load(args):
    """
    parses the dataset
    """
    
    dataset = parser(args.data, args.dataset).parse()
    dataset['partitions'] = partition(dataset['labels'])
    train, test = split(dataset, 1740, args.seed)

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, args.dataset, "splits", str(args.split) + ".pickle")

    if os.path.isfile(file): print("file exists")
    else: 
        S = {"train": train, "test": test}
        with open(file, 'wb') as H: pickle.dump(S, H, protocol=pickle.HIGHEST_PROTOCOL)

    with open(file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    return dataset, train, test



def split(dataset, L, seed):
    """
    splits partitions into equal-sized lists of nodes for train and treats remaining as test
    creates dissimilarity hyperedges

    arguments:
    dataset: a dictionary containing the dataset
    L: size of train

    returns:
    train: numpy array of training indices
    test: numpy array of test indices
    """
    
    p, n = dataset['partitions'], dataset['features'].shape[0]
    c = len(p)  # number of classes
    print("num classes is", c)
    
    assert(L%c == 0), "#labeled nodes (" + str(L) + ") must be divisible by #classes (" + str(c) + ")" 
    l = int(L/c)

    train, splits = [], [0]*c
    random.seed(seed)
    for i in range(c):
        splits[i] = random.sample(list(p[i]), l)
        train = train + splits[i] 
    train = np.array(train)

    test = np.delete(range(n), train)
    train, test = list(train), list(test)
    return train, test



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
        
        name = "_" + self.dataset + "_" + self.data
        function = getattr(self, name, lambda: {})
        return function()



    def _dblp_coauthorship(self):
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

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = np.array(pickle.load(handle), dtype=np.float32)

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



def partition(labels):
    """
    creates partitions based on the label values

    arguments:
    labels: a numpy ndarray of size number of labels (n) X number of classes (c)

    returns:
    list of sets of nodes with each set corresponding to a differente label 
    """
    
    n, c = labels.shape[0], labels.shape[1]
    p = [0]*c   # partitions

    for i in range(n):
        for j in range(c):
            if labels[i][j] == 1:
                if p[j] == 0:
                    p[j]=set()
                p[j].add(i)
                break
    return p