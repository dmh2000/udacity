from keras.models import Sequential
from keras.layers import Conv2D
import math

# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid',
#     activation='relu', input_shape=(200, 200, 1)))
# model.summary()
#
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='valid',
#     activation='relu', input_shape=(200, 200, 1)))
# model.summary()
#
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='valid',
                 activation='relu', input_shape=(200, 200, 1)))
model.summary()

k = 16
f = 3


def parms(k, f, d):
    return k * f * f * d + k


def shape(k, f, s, h, w, pad):
    h = float(h)
    w = float(w)
    s = float(s)
    f = float(f)
    if pad == 'same':
        h = math.ceil(h / s)
        w = math.ceil(w / s)
    elif pad == 'valid':
        h = math.ceil((h - f + 1) / s)
        w = math.ceil((w - f + 1) / s)
    else:
        h = 0
        w = 0;
    return h, w

k = 16
f = 3
s = 1
pad = 'same'
h = 200
w = 100
print(parms(k,f,s))
print(shape(16, 3, 1, 1 , 1, 'same'))
print(shape(16, 3, 1, 1, 1, 'valid'))


from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
    activation='relu', input_shape=(128, 128, 3)))
model.summary()