from __future__ import print_function
import csv
import numpy as np

IMAGE_SIZE = 48
DATA_SIZE = 48 * 48
DATA_FILE = 'fer2013.csv'
BAD_DATA = 'bad_data.txt'

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=10, min_divisor=1e-8):

    """
    __author__ = "David Warde-Farley"
    __copyright__ = "Copyright 2012, Universite de Montreal"
    __credits__ = ["David Warde-Farley"]
    __license__ = "3-clause BSD"
    __email__ = "wardefar@iro"
    __maintainer__ = "David Warde-Farley"
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  
    else:
        X = X.copy()
    if use_std:
        ddof = 1
        if X.shape[1] == 1:
            ddof = 0
        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

def zero_center(data):
    data = data - np.mean(data,axis=0)
    return data

def preproces_data(_list):
    '''
        We apply ZCA to make input zero mean and Identity covariance.
        Covariance structure only represent the linear relationship between variables,
        and that in many cases the dependance structure we are looking isn't expected to be linear.
        So to make sure that our algorithm doesn't waste time on this linear structure, we remove it.

        Global Contrast Normalization computes the mean and standard deviation of the all pixels in the image
        and subtracts the mean from each pixel and divides the pixels by the standard deviation
    '''
    global IMAGE_SIZE
    array = np.asarray(_list)
    data = array.reshape(IMAGE_SIZE, IMAGE_SIZE)
    data = zero_center(data)
    data = zca_whitening(flatten_matrix(data)).reshape(IMAGE_SIZE, IMAGE_SIZE)
    data = global_contrast_normalize(data)
    data = np.rot90(data, 3)
    return data

def load_data():
    global DATA_SIZE, DATA_FILE, BAD_DATA
    train_x = []
    train_y = []
    val_x =[]
    val_y =[]
    bad_training_data = []

    with open(BAD_DATA, "r") as text:
        for line in text:
            bad_training_data.append(int(line))
    number = 0
    reader = csv.reader(open(DATA_FILE))
    # skip the header
    next(reader, None)
    for emotion, image, usage in reader:
        number += 1
        if number not in bad_training_data:
            image_list = [int(pixel) for pixel in image.split()]
            data = preproces_data(image_list)
            emotion = int(emotion)
            data_list = data.reshape(DATA_SIZE).tolist()
            if usage == "Training" or usage == "PublicTest" :
                train_y.append(emotion)
                train_x.append(data_list)
            elif usage == "PrivateTest":
                val_y.append(emotion)
                val_x.append(data_list)

    return train_x, train_y, val_x, val_y
