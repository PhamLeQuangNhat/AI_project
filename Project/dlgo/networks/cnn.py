from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K 

class Model:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((3, 3), input_shape=inputShape))
        model.add(Conv2D(64, (7, 7), padding='valid'))
        model.add(Activation('relu'))
        
        # second ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(64, (5, 5)))
        model.add(Activation('relu'))

        # third ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(64, (5, 5)))
        model.add(Activation('relu'))

        # forth ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(48, (5, 5)))
        model.add(Activation('relu'))
        
        # fith ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(48, (5, 5)))
        model.add(Activation('relu'))

        # sixth ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(32, (5, 5)))
        model.add(Activation('relu'))

        # seventh ZeroPadding => CONV => RELU
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(32, (5, 5)))
        model.add(Activation('relu'))

        # first (and only) FC => RELU 
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
