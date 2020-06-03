import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.datapreprocessing.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
#from dlgo.httpfrontend import get_web_app
from dlgo.networks import cnn

go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

X, y = processor.load_go_data(num_samples=100)

input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
model = Sequential()
network_layers = cnn.layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=20, verbose=1)

deep_learning_bot = DeepLearningAgent(model, encoder)
with h5py.File('./agent/deep_bot.h5', 'w') as outf:
    deep_learning_bot.serialize(outf)



















from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100
encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = small.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',
metrics=['accuracy'])

epochs = 5
batch_size = 128
model.fit_generator(generator=generator.generate(batch_size, num_classes),
                    epochs=epochs,
                    steps_per_epoch=generator.get_num_samples() / batch_size,
                    validation_data=test_generator.generate(
                    batch_size, num_classes),
                    validation_steps=test_generator.get_num_samples() / batch_size,
                    callbacks=[
                    ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')
                    ])
model.evaluate_generator(
generator=test_generator.generate(batch_size, num_classes),
steps=test_generator.get_num_samples() / batch_size)