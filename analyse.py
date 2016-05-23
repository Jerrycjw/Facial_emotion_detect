def select_cate(l,c):
    img = []
    for n in l:
        if n[1] == c:
            img.append(n[0])
    return img
from matplotlib import pyplot as plt
def show_img(num, l):
    f = plt.figure()
    for i in range(num):
        f.add_subplot(num/5,5,i)
        plt.imshow(l[i], interpolation='nearest', cmap='Greys_r')

    plt.show()

