import cPickle
import gzip
import numpy as np
import random
from sklearn import cross_validation
def load_data(path = ['data/face_data_2D.pkl.gz','data/jaffe_2D.pkl.gz','data/KDEF-FACE_2D.pkl.gz']):
    dataset = map(lambda x: cPickle.load(gzip.open(x, 'rb')),path)
    X = dataset[0][0]+dataset[1][0]+dataset[2][0]
    Y = dataset[0][1]+dataset[1][1]+dataset[2][1]
    random.shuffle(X)
    random.shuffle(Y)
    X_train, X_dev, Y_train, Y_dev = cross_validation.train_test_split(X,Y,test_size=0.2)
    X_test, X_val, Y_test, Y_val = cross_validation.train_test_split(X_dev,Y_dev,test_size=0.5)
    return ([X_train,Y_train], [X_val,Y_val], [X_test, Y_test])

if __name__ == '__main__':
    load_data()
