def select_cate(l,c):
    img = []
    for i in xrange(len(l[0])):
        if l[1][i] == c:
            img.append(l[0][i])
    return img
from matplotlib import pyplot as plt
def show_img(num, data, c):
    l = select_cate(data,c)
    f = plt.figure()
    for i in range(num):
        f.add_subplot(num/5,5,i)
        plt.imshow(l[i], interpolation='nearest', cmap='Greys_r')

    plt.show()

import data_loader
train = data_loader.load_data()
show_img(5,train,4)
