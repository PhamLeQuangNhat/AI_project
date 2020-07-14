from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.network.VGGNet import VGGNet
import numpy as np 
num_games_train = 1000000
num_games_test = 20000

go_board_rows, go_board_cols = 19, 19
classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

trainX, trainY = processor.load_go_data('train', num_games_train)
testX, testY = processor.load_go_data('test', num_games_test)
model = VGGNet.build(width=go_board_rows, height=go_board_cols, depth=encoder.num_planes, classes=classes)

opt = SGD(lr=0.03, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

H = model.fit(trainX, trainY,validation_data=(testX,testY), batch_size=64, epochs=200, verbose=1)

deep_learning_bot = DeepLearningAgent(model, encoder)
with h5py.File("H5PY/deep_bot.h5", 'w') as outf:
    deep_learning_bot.serialize(outf)