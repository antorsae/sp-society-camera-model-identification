import argparse
import glob
import numpy as np
import pandas as pd
import random
from scipy.misc import imread, imsave
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, downscale_local_mean
import skimage.exposure
import scipy.ndimage
from keras.optimizers import Adam, Adadelta, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense
from keras.utils import to_categorical
from keras.initializers import Constant
from multi_gpu_keras import multi_gpu_model
from keras import backend as K
from keras.engine.topology import Layer
import skimage
import hickle
from iterm import show_image

import itertools
import re
import os
import sys
import csv
from tqdm import tqdm
import jpeg4py as jpeg
from scipy import signal
import cv2

from keras.applications import *

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-s', '--sub-batch-size', type=int, default=4, help='Number of crops from same image for each batch')
parser.add_argument('-ba', '--batch-acc', type=int, default=1, help='Batch Size for training accumulation, e.g. -b 4')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-p', '--patchsize', type=int, default=299, help='Patch size')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-pp','--preprocessed_input_path', type=str, default='images_preprocessed.hkl', help='Path to preprocessed images')
args = parser.parse_args()

TRAIN_FOLDER = 'train'
TEST_FOLDER  = 'test'
MODEL_FOLDER = 'models'

PATCH_SIZE = args.patchsize
CLASSES = [
  'HTC-1-M7',
  'iPhone-6',     
  'Motorola-Droid-Maxx',
  'Motorola-X',
  'Samsung-Galaxy-S4',
  'iPhone-4s',
  'LG-Nexus-5x', 
  'Motorola-Nexus-6',
  'Samsung-Galaxy-Note3',
  'Sony-NEX-7']

N_CLASSES = len(CLASSES)

def gen(items, batch_size, training=True, inference=False):

  images_cached = {}

  X = np.zeros((batch_size, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
  O = np.zeros((batch_size, 2), dtype=np.float32)

  if not inference:
    y = np.zeros((batch_size), dtype=np.float32)

  load_img   = lambda img_path: jpeg.JPEG(img_path).decode()

  batch_idx = 0
  
  if os.path.isfile(args.preprocessed_input_path):
    images_cached = hickle.load(args.preprocessed_input_path)

  while True:
    if training:
      random.shuffle(items)

    for item in items:

      if not inference:

        class_name = item.split('/')[-2]
        assert class_name in CLASSES
        class_idx = CLASSES.index(class_name)

      if item not in images_cached:

        img = load_img(item)

        #img = skimage.transform.resize(img, (PATCH_SIZE, PATCH_SIZE), mode='reflect').astype(np.float32) / 127.5 - 1.
        #img = img.astype(np.float32) / 127.5 - 1.

        kernel_filter = 1/12. * np.array([\
          [-1,  2,  -2,  2, -1],  \
          [ 2, -6,   8, -6,  2],  \
          [-2,  8, -12,  8, -2],  \
          [ 2, -6,   8, -6,  2],  \
          [-1,  2,  -2,  2, -1]]) 

        #img = cv2.filter2D(img,-1,kernel_filter)

        #images_cached[item] = img

      else:

        img = images_cached[item]

      sub_batch_size = args.sub_batch_size

      assert batch_size % sub_batch_size == 0

      for sub_batch in range(sub_batch_size):
        sx= random.randint(0, img.shape[1] - PATCH_SIZE)
        sy= random.randint(0, img.shape[0] - PATCH_SIZE)

        _img = img[sy:sy+PATCH_SIZE, sx:sx+PATCH_SIZE]
        _img = inception_v3.preprocess_input(_img.astype(np.float32))

        X[batch_idx+sub_batch] = _img
        O[batch_idx+sub_batch] = np.float32([(sx/(img.shape[1] - PATCH_SIZE)), (sy/(img.shape[0] - PATCH_SIZE))])
        y[batch_idx+sub_batch] = class_idx # to_categorical(class_idx, num_classes = N_CLASSES)

      if batch_idx == 0 and False:
        show_image(X[batch_idx])

      batch_idx += sub_batch_size

      if batch_idx == batch_size:

        if not inference:
          #I_O_zipped = list(zip(X, O, y))
          #random.shuffle(I_O_zipped)
          #X[:], O[:], y[:] = zip(*I_O_zipped)
          yield([X, O], [y])
        
        else:   
          I_O_zipped = list(zip(X, O))
          random.shuffle(I_O_zipped)
          X[:], O[:] = zip(*I_O_zipped)

          yield([X, O])

        batch_idx = 0

    #if not os.path.isfile(args.preprocessed_input_path):
    #  hickle.dump(images_cached, args.preprocessed_input_path, mode='w')

class KernelFilter(Layer):

    def __init__(self, **kwargs):
        super(KernelFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.kernel_initializer = Constant(1/12. * np.float32([\
          [-1,  2,  -2,  2, -1],  \
          [ 2, -6,   8, -6,  2],  \
          [-2,  8, -12,  8, -2],  \
          [ 2, -6,   8, -6,  2],  \
          [-1,  2,  -2,  2, -1] ]))

        self.kernel = self.add_weight(
          name='kernel',
          shape=(5,5,3,3),
          initializer=self.kernel_initializer,
          trainable=False)

        super(KernelFilter, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        print(x.shape)
        print(self.kernel)

        return K.conv2d(x, self.kernel, padding='same', data_format='channels_last')

if args.model:
  print("Loading model " + args.model)

  # monkey-patch loss so model loads ok
  # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
  import keras.losses
  import keras.metrics

  model = load_model(args.model, compile=False)
  match = re.search(r'([a-z]+)-epoch(\d+)-.*\.hdf5', args.model)
  model_name = match.group(1)
  last_epoch = int(match.group(2)) + 1

else:
  last_epoch = 0

  input_image = Input(shape=(PATCH_SIZE, PATCH_SIZE, 3))
  crop_zone   = Input(shape=(2,))
  image_filtered = input_image #KernelFilter(input_shape=(PATCH_SIZE, PATCH_SIZE, 3))(input_image)

  classifier_model = InceptionV3(
    include_top=False, 
    weights = 'imagenet', 
    input_shape=(PATCH_SIZE, PATCH_SIZE, 3), 
    pooling='avg', classes=N_CLASSES)

  x = classifier_model(image_filtered)
  x = concatenate([x,crop_zone])
  prediction = Dense(N_CLASSES, activation ="softmax", name="predictions")(x)

  model = Model(inputs=(input_image,crop_zone), outputs=prediction)

  model.summary()

  model_name = 'InceptionV3'

model = multi_gpu_model(model, gpus=args.gpus)

ids = glob.glob(join(TRAIN_FOLDER,'*/*.jpg'))
ids.sort()

ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=42)


opt = Adam(lr=args.learning_rate)
#opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if True:
  metric  = "-val_acc{val_acc:.6f}"
  monitor = 'val_acc'

  save_checkpoint = ModelCheckpoint(
      join(MODEL_FOLDER, model_name+"-epoch{epoch:02d}"+metric+".hdf5"),
      monitor=monitor,
      verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

  reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4, min_lr=1e-7, epsilon = 0.00001, verbose=1, mode='max')

  model.fit_generator(
      generator        = gen(ids_train, args.batch_size),
      steps_per_epoch  = len(ids_train)  // args.batch_size,
      validation_data  = gen(ids_val, args.batch_size, training = False),
      validation_steps = len(ids_val) // args.batch_size,
      epochs = args.max_epoch,
      callbacks = [save_checkpoint, reduce_lr],
      initial_epoch = last_epoch)