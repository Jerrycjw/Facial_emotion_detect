
from matplotlib import pyplot as plt
def show_img(num, l):
    f = plt.figure()
    for i in range(num):
        f.add_subplot(num/5,5,i)
        plt.imshow(l[i], interpolation='nearest', cmap='Greys_r')

    plt.show()

