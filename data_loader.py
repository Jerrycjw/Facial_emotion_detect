import cPickle
import gzip
import numpy as np

def load_data(path = 'data/data.pkl.gz'):
    f = gzip.open(path, 'rb')
    training_data = cPickle.load(f)
    f.close()
    return (training_data)
