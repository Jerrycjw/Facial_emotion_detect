import cPickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('data/data.pkl.gz', 'rb')
    training_data = cPickle.load(f)
    f.close()
    return (training_data)
