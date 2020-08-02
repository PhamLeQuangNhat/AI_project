from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
import h5py
from dlgo.network.VGGNet import VGGNet
import numpy as np
from dlgo.agent.predict import DeepLearningAgent

num_games_train = 100
num_games_test = 5

go_board_rows, go_board_cols = 19, 19
classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

trainX, trainY = processor.load_go_data('train', num_games_train)
#testX, testY = processor.load_go_data('test', num_games_test)
model = VGGNet.build(width=go_board_rows, height=go_board_cols, depth=encoder.num_planes, classes=classes)

#opt = SGD(lr=0.03, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="sgd",
                metrics=["accuracy"])

H = model.fit(trainX, trainY, batch_size=64, epochs=2, verbose=1)

deep_learning_bot = DeepLearningAgent(model, encoder)
with h5py.File("H5PY/deep_bot.h5", 'w') as outf:
    deep_learning_bot.serialize(outf)
