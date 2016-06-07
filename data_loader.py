import cPickle
import gzip
import numpy as np
import random

def load_data(path = ['data/face_data.pkl.gz','data/jaffe.pkl.gz','data/KDEF-FACE.pkl.gz']):
    fs = map(lambda x: gzip.open(x, 'rb'),path)
    training_data = []
    validation_data = []
    test_data = []
    for f in fs:
        training, validation, test = cPickle.load(f)
        for x in training: training_data[0].append(x[0]); training_data[1].append(x[1]);
        for x in validation: validation_data[0].append(x[0]); validation_data[1].append(x[1]);
        for x in test: test_data[0].append(x[0]); test_data[1].append(x[1]);
        f.close()
    random.shuffle(training_data)
    return (training_data, validation_data, test_data)

