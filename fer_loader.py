import csv
import cPickle
import numpy as np
import gzip
def transform(path = 'data/fer2013/fer2013.csv'):
    pairs = []
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            label = row[0]
            image = row[1].split(' ')
            X = np.array(image,dtype='uint8').reshape(48,48)
            pairs.append ((X, label))
    train_num = 28709
    test_num = 3589
    train_pairs = pairs[:train_num]
    public_test_pairs = pairs[train_num:train_num+test_num]
    private_test_pairs = pairs[train_num+test_num:]
    data_train = [list(d) for d in zip(*train_pairs)]
    data_pbtest = [list(d) for d in zip(*public_test_pairs)]
    data_pvtest =[list(d) for d in zip(*private_test_pairs)]

    cPickle.dump(data_train, gzip.open('data/fer_dataset_train.pkl.gz', 'w'))
    cPickle.dump(data_pbtest, gzip.open('data/fer_dataset_pbtest.pkl.gz', 'w'))
    cPickle.dump(data_pvtest, gzip.open('data/fer_dataset_pvtest.pkl.gz', 'w'))

def load_data():
    train = cPickle.load(gzip.open('data/fer_dataset_train.pkl.gz', 'rb'))
    valid = cPickle.load(gzip.open('data/fer_dataset_pvtest.pkl.gz', 'rb'))
    test = cPickle.load(gzip.open('data/fer_dataset_pbtest.pkl.gz', 'rb'))
    return (train,valid,test)

if __name__ == '__main__':
    transform()