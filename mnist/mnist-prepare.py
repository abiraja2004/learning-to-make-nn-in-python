import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST

	Note that you first need to download files
	http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	and unpack them
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    labels += 1
    labels  = labels.flatten()

    images  = images.astype(np.float16)
    images  = images.reshape(( images.shape[0], -1 )) # FLATTENING for plain DNN

    return images, labels


testing, lab_testing = load_mnist('testing')
trainin, lab_trainin = load_mnist('training')

mean = trainin.astype(np.float64).mean()
std  = trainin.astype(np.float64).std()
print(mean, std)

np.save	(	'mnist.trainin'
		,	trainin
		,	allow_pickle = False
		)
np.save	(	'mnist.testing'
		,	testing
		,	allow_pickle = False
		)

np.save	(	'mnist.lab_testing'
		,	lab_testing
		,	allow_pickle = False
		)
np.save	(	'mnist.lab_trainin'
		,	lab_trainin
		,	allow_pickle = False
		)
