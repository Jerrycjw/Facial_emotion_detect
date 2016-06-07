import cPickle
import gzip
import numpy as np
import random

def load_data(path = ['data/face_data.pkl.gz','data/jaffe.pkl.gz','KDEF-FACE.pkl.gz']):
    fs = map(lambda path: gzip.open(path, 'rb'))
    training_data = []
    validation_data = []
    test_data = []
    for f in fs:
        training, validation, test = cPickle.load(f)
        training_data.append(training)
        validation_data.append(validation)
        test_data.append(test)
        f.close()
    random.shuffle(training_data)
    return (training_data, validation_data, test_data)

