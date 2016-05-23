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
i = 0

for dir in label_dirs:
    img_path = 'data/cohn-kanade-images/'+dir.split('data/Emotion/')[1]
    label_list = glob.glob(dir + '/*.txt')
    try:
        file = open(label_list[0])
        y = int(file.readline().rstrip().lstrip()[0])
        img_list = glob.glob(img_path + '/*.png')
        files = img_list[-file_num:]
        imgs = map(lambda x: misc.imread(x,'L'), files)
        for x in imgs:
            pairs.append((x, y))
        print "read file number: {0}".format(i)
        i = i+1
    except:
        pass
random.shuffle(pairs)
data = [list(d) for d in zip(*pairs)]
print("Saving expanded data. This may take a few minutes.")
f = gzip.open("data/data2.pkl.gz", "w")
cPickle.dump(data[1], f)
print("Successfully")
f.close()