from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu',input_shape=(1,64,64)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model



if __name__ == '__main__':
    import data_loader
    train, valid, test = data_loader.load_data(['data/face_data_2D.pkl.gz','data/jaffe_2D.pkl.gz','data/KDEF-FACE_2D.pkl.gz'])
    model = VGG_16()
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.005, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    X_train = np.array(train[0]).reshape(len(train[0]),1,64,64)
    X_valid =  np.array(valid[0]).reshape(len(valid[0]),1,64,64)
    X_test = np.array(test[0]).reshape(len(test[0]),1,64,64)
    from keras.utils.np_utils import to_categorical
    Y_train = to_categorical(np.array(train[1]))
    Y_valid = to_categorical(np.array(valid[1]))
    Y_test = to_categorical(np.array(test[1]))
    train_out = model.fit(X_train,Y_train,nb_epoch=100,batch_size=30,validation_data=(X_valid,Y_valid))
    test_out = model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)