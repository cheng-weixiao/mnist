# The file used to generate mnist.pkl.gz

import numpy as np
import pickle
import gzip
from matplotlib import pyplot
import matplotlib as mpl
def load_image(images, labels, img_num, img_size=28*28):
    """return data with shape (samples, img_size+1), last column is labels"""
    
    data = np.zeros((img_num, img_size+1), dtype=np.uint8)
    with gzip.open(images) as f_images, gzip.open(labels) as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in range(img_num):
            for j in range(img_size+1):
                if j == img_size:
                    data[i,j] = ord(f_labels.read(1))
                else:
                    data[i,j] = ord(f_images.read(1))
                
    return data

def loadmnist():
    #Load mnist
    train = load_image('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz', 60000)

    #Show the figure of image i
    i=0;
    image = train[i,:-1].reshape((28, 28))
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    pyplot.show()

    #Split the training set to three groups
    perm = np.random.permutation(train.shape[0])
    X1 = train[perm[:20000],:]
    X2 = train[perm[20000:40000],:]
    X3 = train[perm[40000:60000],:]
    test = load_image('data/t10k-images-idx3-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz', 10000)
    mnist = (X1, X2, X3, test)

    with gzip.open('mnist.pkl.gz', 'wb') as output:
        pickle.dump(mnist, output, -1)


loadmnist()