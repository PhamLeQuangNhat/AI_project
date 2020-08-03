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

        model.add(ZeroPadding2D((3, 3), input_shape=input_shape, data_format='channels_first'))
	model.add(Conv2D(64, (7, 7), padding='valid', data_format='channels_first'))
	model.add(Activation('relu'))
	
	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(64, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(64, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(48, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(48, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(32, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(ZeroPadding2D((2, 2), data_format='channels_first'))
	model.add(Conv2D(32, (5, 5), data_format='channels_first'))
	model.add(Activation('relu'))

        # Block #6: FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        # Block #7: FC => RELU layers
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the network architecture
        return model    

