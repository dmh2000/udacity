import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(42)
# ========================================================
# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)
# ========================================================
print(x_train[0],len(x_train[0]))
print(y_train[0])
# ========================================================
# One-hot encoding the output into vector mode, each of length 1000
num_words = 1000
tokenizer = Tokenizer(num_words=num_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])
# ========================================================
# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# ========================================================
from keras import optimizers

# Build the model architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(num_words,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

# Compile the model using a loss function and an optimizer.
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.summary()

# ========================================================


# Run the model. Feel free to experiment with different batch sizes and number of epochs.
model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=2)
print('done')

score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
