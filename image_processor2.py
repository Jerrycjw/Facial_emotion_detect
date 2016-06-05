import numpy as np
import glob
import cPickle
import gzip
from scipy import misc
import random
# This script is used for deal with jaffe dataset
categories ={'NE':0, 'AN':1, 'DI':3, 'FE':4, 'HA':5, 'SA':6, 'SU':7}
image_path='data/jaffe/*.tiff'
pairs = []
index = 0
image_list = glob.glob(image_path)
imgs = map(lambda x: misc.imread(x,'L'), image_list)
label = map(lambda x: categories[x[14:16]], image_list)
imgs_face_small = map(lambda x: misc.imresize(x,(64,64)).reshape(64*64),imgs)
for i in xrange(len(image_list)):
    pairs.append((imgs_face_small[i],label[i]))
training_data = [list(d) for d in zip(*pairs[:int(len(pairs)*0.8)])]
validation_data = [list(d) for d in zip(*pairs[int(len(pairs)*0.8):int(len(pairs)*0.9)])]
test_data = [list(d) for d in zip(*pairs[int(len(pairs)*0.9):])]

f = gzip.open('data/jaffe.pkl.gz', 'w')
cPickle.dump((training_data, validation_data, test_data), f)
