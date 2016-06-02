#coding=utf-8
import numpy as np
import os
import sys
import timeit
import glob
import cPickle
import gzip
from scipy import misc
import random
import numpy
import theano
theano.config.floatX = 'float32'
import sklearn
import theano.tensor as T
from sklearn import cross_validation
from logistic_sgd import LogisticRegression, load_data
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from mlp import HiddenLayer
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import sigmoid
from image_processor import LeNetConvPoolLayer
import six.moves.cPickle as pickle
from logistic_sgd import LogisticRegression

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    layer0 = pickle.load(open('best_model_layer0.pkl'))
    layer2 = pickle.load(open('best_model_layer2.pkl'))
    layer3 = pickle.load(open('best_model_layer3.pkl'))

    # compile a predictor function

    """
    predict_model1 = theano.function(
        inputs=[layer0.input],
        outputs=layer0.output)
    predict_model2 = theano.function(
        inputs=[layer2.input],
        outputs=layer2.output
    )
    predict_model3 = theano.function(
        inputs=[layer3.input],
        outputs=layer3.y_pred
    )
    """
    predict_model = theano.function(
        inputs=[layer0.input],
        outputs=[layer3.y_pred]
    )






    # We can test it on some examples from test test
    f = gzip.open('/Users/yuanjun/Desktop/DeepLearning/data/data.pkl.gz', 'rb')
    dataset= cPickle.load(f)

    #dataset='mnist.pkl.gz'
    #datasets = load_data(dataset)

    x1 = dataset[0]
    #test_set_x = test_set_x.get_value()

    x1 = np.array(x1).reshape(len(x1),64*64)
    x1 = x1.astype(np.float32)
    test_set_x = theano.shared(numpy.asarray(x1,dtype=theano.config.floatX))


    """
    predicted_values_1 = predict_model1(test_set_x[:10])
    predicted_values_2 = predict_model2(predicted_values_1)
    predicted_values_3 = predict_model3(predicted_values_2)
    """
    predicted_values = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

predict()