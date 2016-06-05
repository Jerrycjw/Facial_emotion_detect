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
import six.moves.cPickle as pickle








#卷积神经网络的一层，包含：卷积+下采样两个步骤
#算法的过程是：卷积-》下采样-》激活函数
class LeNetConvPoolLayer(object):

    #image_shape是输入数据的相关参数设置  filter_shape本层的相关参数设置
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        3、input: 输入特征图数据，也就是n幅特征图片

        4、参数 filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        num of filters：是卷积核的个数，有多少个卷积核，那么本层的out feature maps的个数
        也将生成多少个。num input feature maps：输入特征图的个数。
        然后接着filter height, filter width是卷积核的宽高，比如5*5,9*9……
        filter_shape是列表，因此我们可以用filter_shape[0]获取卷积核个数

        5、参数 image_shape: (batch size, num input feature maps,
                             image height, image width)，
         batch size：批量训练样本个数 ，num input feature maps：输入特征图的个数
         image height, image width分别是输入的feature map图片的大小。
         image_shape是一个列表类型，所以可以直接用索引，访问上面的4个参数，索引下标从
         0~3。比如image_shape[2]=image_heigth  image_shape[3]=num input feature maps

        6、参数 poolsize: 池化下采样的的块大小，一般为(2,2)
        """

        assert image_shape[1] == filter_shape[1]#判断输入特征图的个数是否一致，如果不一致是错误的
        self.input = input

        # fan_in=num input feature maps *filter height*filter width
        #numpy.prod(x)函数为计算x各个元素的乘积
        #也就是说fan_in就相当于每个即将输出的feature  map所需要链接参数权值的个数
        fan_in = numpy.prod(filter_shape[1:])
        # fan_out=num output feature maps * filter height * filter width
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # 把参数初始化到[-a,a]之间的数，其中a=sqrt(6./(fan_in + fan_out)),然后参数采用均匀采样
        #权值需要多少个？卷积核个数*输入特征图个数*卷积核宽*卷积核高？这样没有包含采样层的链接权值个数
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # b为偏置，是一维的向量。每个输出特征图i对应一个偏置参数b[i]
        #,因此下面初始化b的个数就是特征图的个数filter_shape[0]
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # 卷积层操作，函数conv.conv2d的第一个参数为输入的特征图，第二个参数为随机出事化的卷积核参数
        #第三个参数为卷积核的相关属性，输入特征图的相关属性
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # 池化操作，最大池化
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        #激励函数，也就是说是先经过卷积核再池化后，然后在进行非线性映射
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # 保存参数
        self.params = [self.W, self.b]
        self.input = input
    def getstate(self):
        return self.W, self.b


def evaluate_lenet5(learning_rate=0.005, n_epochs=5,data = None,nkerns= 64, batch_size=30):

    x_val=data[0]
    y_val=data[1]

    print len(x_val)

    x1=[]
    y1=[]


    #for i in range(len(x_val)):
        #if len(x_val[i]) == 490 and len(x_val[i][0]) == 640:
            #x1.append(x_val[i])
            #y1.append(y_val[i]-1)
            #if len(x1) == 80:
                #break



    for i in range(len(x_val)):
            x1.append(x_val[i])
            y1.append(y_val[i]-1)
            if len(x1) == 500:
                break

    print len(x1)
    print len(y1)

    x1 = np.array(x1).reshape(500,64*64)
    x1 = x1.astype(np.float32)


    x_train, x2, y_train, y2 = cross_validation.train_test_split(x1,y1,test_size=0.4,random_state=0)
    x_valid, x_test, y_valid, y_test = cross_validation.train_test_split(x2,y2,test_size=0.5,random_state=0)

    x_train2 = theano.shared(numpy.asarray(x_train,dtype=theano.config.floatX))
    y_train2 = theano.shared(numpy.asarray(y_train,dtype=theano.config.floatX))
    x_valid2 = theano.shared(numpy.asarray(x_valid,dtype=theano.config.floatX))
    y_valid2 = theano.shared(numpy.asarray(y_valid,dtype=theano.config.floatX))
    x_test2 = theano.shared(numpy.asarray(x_test,dtype=theano.config.floatX))
    y_test2 = theano.shared(numpy.asarray(y_test,dtype=theano.config.floatX))

    y_train2 = T.cast(y_train2, 'int32')
    y_test2 = T.cast(y_test2, 'int32')
    y_valid2 = T.cast(y_valid2, 'int32')

    print len(x_train)
    print len(y_train)

    rng = numpy.random.RandomState(23455)

    n_train_batches = len(y_train)/batch_size
    n_valid_batches = len(y_valid)/batch_size
    n_test_batches = len(y_test)/batch_size
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are p

    layer0_input = x.reshape((batch_size, 1, 64, 64))

    '''构建第一层网络：
    image_shape：输入大小为490*640的特征图，batch_size个训练数据，每个训练数据有1个特征图
    filter_shape：卷积核个数为nkernes=64，因此本层每个训练样本即将生成64个特征图
    经过卷积操作，图片大小变为(490-7+1 , 640-7+1) = (484, 634)
    经过池化操作，图片大小变为 (484/2, 634/2) = (242, 317)
    最后生成的本层image_shape为(batch_size, nklearn, 242, 317)'''

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 64, 64),
        filter_shape=(nkerns, 1, 7, 7),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns * 7 * 7),
    # (100, 64*7*7) with the default values.
    layer2_input = layer0.output.flatten(2)

    '''全链接：输入layer2_input是一个二维的矩阵，第一维表示样本，第二维表示上面经过卷积下采样后
    每个样本所得到的神经元，也就是每个样本的特征，HiddenLayer类是一个单层网络结构
    下面的layer2把神经元个数由800个压缩映射为500个'''
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns * 29 * 29,
        n_out=500,
        activation=T.tanh
    )

    layer2.output = dropout_layer(layer2.output,0.5)

    # 最后一层：逻辑回归层分类判别，把500个神经元，压缩映射成10个神经元，分别对应于手写字体的0~9
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=7)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            y: y_test2[index * batch_size: (index + 1) * batch_size],
            x: x_test2[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: x_valid2[index * batch_size: (index + 1) * batch_size],
            y: y_valid2[index * batch_size: (index + 1) * batch_size]
        }
    )

    #把所有的参数放在同一个列表里，可直接使用列表相加
    params = layer3.params + layer2.params  + layer0.params

    #梯度求导
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: x_train2[index * batch_size: (index + 1) * batch_size],
            y: y_train2[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.2  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
    #while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):#每一批训练数据

            cost_ij = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    with open('param0.pkl', 'wb') as f0:
                        pickle.dump(layer0.getstate(), f0)
    f0.close()
    with open('param2.pkl', 'wb') as f2:
                        pickle.dump((layer2.W,layer2.b), f2)
    f2.close()
    with open('param3.pkl', 'wb') as f3:
                        pickle.dump(layer3.getstate(), f3)
    f3.close()

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def getstate(self):
        return self.W, self.b

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)





def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

if __name__ == '__main__':
    f = gzip.open('data/data.pkl.gz', 'rb')
    dataset = cPickle.load(f)
    evaluate_lenet5(data=dataset)