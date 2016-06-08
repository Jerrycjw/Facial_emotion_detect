import cPickle
import gzip
import numpy as np
import random
from sklearn import cross_validation
def load_data(path ='dataset_2D_64.pkl.gz'):
    data = cPickle.load(gzip.open(path))
    X = data[0]
    Y = data[1]
    X_train, X_dev, Y_train, Y_dev = cross_validation.train_test_split(X,Y,test_size=0.2)
    X_test, X_val, Y_test, Y_val = cross_validation.train_test_split(X_dev,Y_dev,test_size=0.5)
    return ([X_train,Y_train], [X_val,Y_val], [X_test, Y_test])

if __name__ == '__main__':
    load_data()
