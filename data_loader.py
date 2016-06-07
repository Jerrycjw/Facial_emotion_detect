import cPickle
import gzip
import numpy as np
import random

def load_data(path = ['data/face_data.pkl.gz','data/jaffe.pkl.gz','data/KDEF-FACE.pkl.gz']):
    fs = map(lambda x: gzip.open(x, 'rb'),path)
    training_data, validation_data, test_data = cPickle.load(fs[0])

    for f in fs[1:]:
        print(len(training_data[0]))
        training, validation, test = cPickle.load(f)
        for x in xrange(len(training[0])): training_data[0].append(training[0][x]); training_data[1].append(training[1][x]);
        for x in xrange(len(validation[0])): validation_data[0].append(validation[0][x]); validation_data[1].append(validation[1][x]);
        for x in xrange(len(validation[0])): test_data[0].append(test[0][x]); test_data[1].append(test[1][x]);
        f.close()
    return (training_data, validation_data, test_data)

if __name__ == '__main__':
    load_data()