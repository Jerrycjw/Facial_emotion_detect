import numpy as np
import glob
import cPickle
import gzip
from scipy import misc
import random


image_dir_path='data/cohn-kanade-images/*/*'
label_dir_path='data/Emotion/*/*'
file_num=6
image_dirs = glob.glob(image_dir_path)
label_dirs = glob.glob(label_dir_path)
pairs = []
index = 0

for dir in label_dirs:
    try:
        img_path = 'data/cohn-kanade-images/'+dir.split('data/Emotion/')[1]
        landmark_path = 'data/Landmarks/'+dir.split('data/Emotion/')[1]
        label_list = glob.glob(dir + '/*.txt')
        file = open(label_list[0])
        y = int(file.readline().rstrip().lstrip()[0])
        img_list = glob.glob(img_path + '/*.png')
        landmark_list = glob.glob(landmark_path + '/*.txt')
        ## sample data by step size
        length = len(img_list)
        step = length/file_num
        files_img = [img_list[i*step] for i in xrange(file_num)]
        files_landmark = [landmark_list[i * step] for i in xrange(file_num)]
        ## read images
        imgs = map(lambda x: misc.imread(x,'L'), files_img)
        ## process landmark data
        landmarks = [ [line.rstrip().split('   ')[-2:] for line in open(x)] for x in files_landmark]
        int_lm = [map(lambda u : map(lambda x: int(float(x)),u),lm) for lm in landmarks]
        ## get face from landmark
        imgs_face = []
        for ii in xrange(file_num):
            l_x = np.asarray([x[0] for x in int_lm[ii]])
            l_y = np.asarray([x[1] for x in int_lm[ii]])
            img_face = imgs[ii]
            imgs_face.append(img_face[l_y.min():l_y.max(), l_x.min():l_x.max()])
        ## resize image to 64*64
        imgs_face_small = map(lambda x: misc.imresize(x,(64,64)),imgs_face)
        for n in xrange(file_num):
            pairs.append((imgs_face_small[n], y, int_lm[n]))
        print "read file number: {0}".format(index)
        index += 1
    except:
        pass

random.shuffle(pairs)
data = [list(d) for d in zip(*pairs)]

f = gzip.open('data/data.pkl.gz', 'rb')
cPickle.dump(data,f)
