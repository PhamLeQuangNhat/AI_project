# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class VGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        # Block #1: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3,3), pad=(1,1), input_shape=inputShape, name="conv1_1"))
        model.add(Activation("prelu",name="act1_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn1_1"))
        model.add(Conv2D(64, (3,3), pad=(1,1), name="conv1_2"))
        model.add(Activation("prelu",name="act1_2"))
        model.add(BatchNormalization(axis=chanDim, name="bn1_2"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1'))
        model.add(Dropout(0.25))

        # Block #2: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3,3), pad=(1,1), name="conv2_1"))
        model.add(Activation("prelu",name="act2_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn2_1"))
        model.add(Conv2D(128, (3,3), pad=(1,1), name="conv2_2"))
        model.add(Activation("prelu",name="act2_2"))
        model.add(BatchNormalization(axis=chanDim, name="bn2_2"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool2'))
        model.add(Dropout(0.25))

        # Block #3: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(256, (3,3), pad=(1,1), name="conv3_1"))
        model.add(Activation("prelu",name="act3_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn3_1"))
        model.add(Conv2D(256, (3,3), pad=(1,1), name="conv3_2"))
        model.add(Activation("prelu",name="act3_2"))
        model.add(BatchNormalization(axis=chanDim, name="bn3_2"))
        model.add(Conv2D(256, (3,3), pad=(1,1), name="conv3_3"))
        model.add(Activation("prelu",name="act3_3"))
        model.add(BatchNormalization(axis=chanDim, name="bn3_3"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool3'))
        model.add(Dropout(0.25))

        # Block #4: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv4_1"))
        model.add(Activation("prelu",name="act4_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn4_1"))
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv4_2"))
        model.add(Activation("prelu",name="act4_2"))
        model.add(BatchNormalization(axis=chanDim, name="bn4_2"))
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv4_3"))
        model.add(Activation("prelu",name="act4_3"))
        model.add(BatchNormalization(axis=chanDim, name="bn4_3"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool4'))
        model.add(Dropout(0.25))

        # Block #5: (CONV => RELU) * 3 => POOL
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv5_1"))
        model.add(Activation("prelu",name="act5_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn5_1"))
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv5_2"))
        model.add(Activation("prelu",name="act5_2"))
        model.add(BatchNormalization(axis=chanDim, name="bn5_2"))
        model.add(Conv2D(512, (3,3), pad=(1,1), name="conv5_3"))
        model.add(Activation("prelu",name="act5_3"))
        model.add(BatchNormalization(axis=chanDim, name="bn5_3"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool5'))
        model.add(Dropout(0.25))

        # Block #6: FC => RELU layers
        model.add(Flatten(name="flatten"))
        model.add(Dense(2048,name="fc1"))
        model.add(Activation("prelu",name="act6_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn6_1"))
        model.add(Dropout(0.25))

        # Block #7: FC => RELU layers
        model.add(Dense(1024,name="fc1"))
        model.add(Activation("prelu",name="act7_1"))
        model.add(BatchNormalization(axis=chanDim, name="bn7_1"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the network architecture
        return model    

