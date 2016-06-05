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

    # We can test it on some examples from test test
    try:
        with open('data/test.pkl.gz', 'rb') as f_in:
            x1 = cPickle.load(f_in)
            test_set_x = theano.shared(numpy.asarray(x1, dtype=theano.config.floatX))
            f_in.close()
    except:
        f = gzip.open('data/data.pkl.gz', 'rb')
        f_out =  gzip.open('data/test.pkl.gz','w')
        dataset = cPickle.load(f)

        # dataset='mnist.pkl.gz'
        # datasets = load_data(dataset)

        x = dataset[0]
        # test_set_x = test_set_x.get_value()
        x = np.array(x).reshape(len(x), 64 * 64)
        x1 = x[0:30].reshape(30, 1, 64, 64)
        x1 = x1.astype(np.float32)
        test_set_x = theano.shared(numpy.asarray(x1, dtype=theano.config.floatX))
        cPickle.dump(x1,f_out)
        f.close()
        f_out.close()
    # load the saved model
    #layer0 = pickle.load(open('best_model_layer0.pkl'))
    #layer2 = pickle.load(open('best_model_layer2.pkl'))
    #layer3 = pickle.load(open('best_model_layer3.pkl'))

    # compile a predictor function

    """
    predict_model1 = theano.function(
        inputs=[layer0.input],
        outputs=[layer0.output])
    predict_model2 = theano.function(
        inputs=[layer2.input],
        outputs=[layer2.output]
    )
    predict_model3 = theano.function(
        inputs=[layer3.input],
        outputs=[layer3.y_pred]
    )
    """
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are p
    layer0_w, layer0_b = pickle.load(open('param0.pkl'))
    layer2_w, layer2_b = pickle.load(open('param2.pkl'))
    layer3_w, layer3_b = pickle.load(open('param3.pkl'))
    nkerns = 64
    batch_size = 30
    layer0_input = x.reshape((batch_size, 1, 64, 64))
    rng = numpy.random.RandomState(23455)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 64, 64),
        filter_shape=(nkerns, 1, 7, 7),
        poolsize=(2, 2)
    )
    layer0.W = layer0_w
    layer0.b = layer0_b
    layer2_input = layer0.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns * 29 * 29,
        n_out=500,
        activation=T.tanh
    )
    layer2.W = layer2_w
    layer2.b = layer2_b
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=7)
    layer3.W = layer3_w
    layer3.b = layer3_b
    """
    layer0.input = test_set_x
    layer2_input = layer0.output.flatten(2)
    layer2.input = layer2_input
    layer3.input = layer2.output
    """

    predict_model = theano.function(
        inputs=[layer0_input],
        outputs=layer3.y_pred)




    predicted_values = predict_model(x1)
    print("Predicted values for the first 30 examples in test set:")
    print(predicted_values)
if __name__ == '__main__':
    predict()