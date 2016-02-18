'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 6 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt
import glob

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

if __name__ == '__main__':

    # images and corresponding classes
    X = []
    y = []

    # class folders
    class_folders = sorted(glob.glob('GTSRB_subset_2/*'))
    
    for i,folder in enumerate(class_folders):
        name_list = glob.glob(folder+'/*')
        for name in name_list:
            image = plt.imread(name)

            # convert to double
            image = image.astype(np.double)
            # normalize pixel values to the range 0-1
            image /= 255.0
            # transpose
            image = np.transpose(image)

            # save data
            X.append(image)
            y.append(i)

    X = np.array(X)
    y = np.array(y)

    # one hot encoding
    y = np_utils.to_categorical(y, 2)

    # convolutional filter size
    w, h = 3, 3

    # cnn structure
    model = Sequential()

    model.add(Convolution2D(32, w, h, border_mode='same', 
                            input_shape=(3,64,64)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, w, h, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # build the model and use sgd with default parameters
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    
    # train and test
    model.fit(X, y, batch_size=32, nb_epoch=20, show_accuracy=True,
              validation_split=0.1, shuffle=True)
