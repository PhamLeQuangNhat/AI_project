# import the necessary packages
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D

class VGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (depth, height, width)

        # Block #1: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3,3), padding="same", input_shape=inputShape, data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu")) 
        model.add(ZeroPadding2D(padding=2,data_format='channels_first'))
        model.add(Dropout(0.25))

        # Block #2: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3,3), padding="same", data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3,3), padding="same", data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(ZeroPadding2D(padding=2,data_format='channels_first'))
        model.add(Dropout(0.25))

        # Block #3: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(256, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3,3), padding="same", data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(ZeroPadding2D(padding=2, data_format='channels_first'))
        model.add(Dropout(0.25))

        # Block #4: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(ZeroPadding2D(padding=2, data_format='channels_first'))
        model.add(Dropout(0.25))

        # Block #5: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3,3), padding="same",data_format='channels_first'))
        model.add(Activation("relu"))
        model.add(ZeroPadding2D(padding=2, data_format='channels_first'))
        model.add(Dropout(0.25))

        # Block #6: FC => RELU layers
        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        # Block #7: FC => RELU layers
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the network architecture
        return model    

