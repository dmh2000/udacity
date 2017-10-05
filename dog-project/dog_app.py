from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

import random

random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

import cv2
import matplotlib.pyplot as plt

# % matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# display the image, along with bounding box
# plt.imshow(cv_rgb)
# plt.show()


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

# Test the performance of the face_detector algorithm
# on the images in human_files_short and dog_files_short.
# humans = 0
# for h in human_files_short:
#     isface = face_detector(h)
#     if isface:
#         humans += 1
#
# dogs = 0
# for d in dog_files_short:
#     isface = face_detector(d)
#     if isface:
#         dogs += 1
#
# print(humans, dogs)


# import dlib
# from skimage import io
#
#
# def dlib_face(img_path):
#     detector = dlib.get_frontal_face_detector()
#     img = io.imread(img_path)
#     dets = detector(img, 1)
#     return len(dets) > 0
#
#
# humans = 0
# for h in human_files_short:
#     isface = dlib_face(h)
#     if isface:
#         humans += 1
#
# dogs = 0
# for d in dog_files_short:
#     isface = dlib_face(d)
#     if isface:
#         dogs += 1
# print(humans,dogs)

# ===================================================================
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
print("1-")
# ===================================================================
from keras.preprocessing import image
from tqdm import tqdm


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

print("2-")
# ===================================================================
from keras.applications.resnet50 import preprocess_input, decode_predictions


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

print("3-")
# ===================================================================
# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

print("4-")
# ===================================================================
# Test the performance of the dog_detector function
# on the images in human_files_short and dog_files_short.
# Test the performance of the face_detector algorithm
# on the images in human_files_short and dog_files_short.
humans = 0
for h in human_files_short:
    isface = dog_detector(h)
    if isface:
        humans += 1

dogs = 0
for d in dog_files_short:
    isface = dog_detector(d)
    if isface:
        dogs += 1

print("percentage of dogs found in human data (dog_detector) ", end='')
print(float(humans) / len(human_files_short))
print("percentage of dogs found in dog data  (dog_detector) ", end='')
print(float(dogs) / len(dog_files_short))
print("5-")
# ===================================================================

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

print("6-")

# ===================================================================


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))

model.summary()


bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19  = bottleneck_features['test']

# bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
# train_Resnet50  = bottleneck_features['train']
# valid_Resnet50  = bottleneck_features['valid']
# test_Resnet50  = bottleneck_features['test']
#
# bottleneck_features = np.load('bottleneck_features/DogInceptionData.npz')
# train_Inception  = bottleneck_features['train']
# valid_Inception  = bottleneck_features['valid']
# test_Inception  = bottleneck_features['test']
#
# bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
# train_Xception  = bottleneck_features['train']
# valid_Xception  = bottleneck_features['valid']
# test_Xception  = bottleneck_features['test']


VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_model.add(Dense(133, activation='softmax'))
VGG19_model.summary()

# Resnet50_model = Sequential()
# Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
# Resnet50_model.add(Dense(133, activation='softmax'))
# Resnet50_model.summary()
#
# Inception_model = Sequential()
# Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:]))
# Inception_model.add(Dense(133, activation='softmax'))
# Inception_model.summary()
#
# Xception_model = Sequential()
# Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
# Xception_model.add(Dense(133, activation='softmax'))
# Xception_model.summary()

from time import t