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
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten
from keras.utils import to_categorical, Sequence
from keras.initializers import Constant
from keras.applications import *

from multi_gpu_keras import multi_gpu_model
from keras import backend as K
from keras.engine.topology import Layer
import skimage
import pickle
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
import math


parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=100, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-s', '--sub-batch-size', type=int, default=1, help='Number of crops from same image for each batch')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-cs', '--crop-size', type=int, default=299, help='Crop size')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-pp','--preprocessed_input_path', type=str, default='images_preprocessed', help='Path to preprocessed images')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')

args = parser.parse_args()

args.preprocessed_input_path += '-kf.pkl' if args.kernel_filter else '.pkl'

TRAIN_FOLDER = 'train'
TEST_FOLDER  = 'test'
MODEL_FOLDER = 'models'

CROP_SIZE = args.crop_size
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
load_img   = lambda img_path: jpeg.JPEG(img_path).decode()

from PIL import Image
from io import BytesIO

def random_manipulation(img):

    manipulation = random.choice(['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2'])

    if manipulation.startswith('jpg'):
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=70 if manipulation=='jpeg70' else 90)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = 0.8 if manipulation == 'gamma0.8' else 1.2
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        img = img/255.0
        img = cv2.pow(img, gamma)
        im_decoded = np.uint8(img*255)

    return im_decoded

def preprocess_image(img):
        # find `preprocess_input` function specific to the classifier
    classifier_to_module = { 
        'NASNetLarge'       : 'nasnet',
        'NASNetMobile'      : 'nasnet',
        'DenseNet121'       : 'densenet',
        'DenseNet161'       : 'densenet',
        'DenseNet201'       : 'densenet',
        'InceptionResNetV2' : 'inception_resnet_v2',
        'InceptionV3'       : 'inception_v3',
        'MobileNet'         : 'mobilenet',
        'ResNet50'          : 'resnet50',
        'VGG16'             : 'vgg16',
        'VGG19'             : 'vgg19',
        'Xception'          : 'xception',
    }

    classifier_module_name = classifier_to_module[args.classifier]
    preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
    return preprocess_input_function(img.astype(np.float32))

def get_size(obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
                seen = set()
        obj_id = id(obj)
        if obj_id in seen:
                return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
        return size


def gen(items, batch_size, training=True, inference=False):

    images_cached = {}

    # X holds image crops
    X = np.zeros((batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    # O holds whether the image has been manipulated or not
    O = np.zeros((batch_size, 1), dtype=np.float32)

    if not inference:
        y = np.zeros((batch_size), dtype=np.int64)

    batch_idx = 0
    
    if os.path.isfile(args.preprocessed_input_path) and training:
        print("Loading preprocessed images from: "+ args.preprocessed_input_path)
        images_cached = pickle.load(open(args.preprocessed_input_path, "rb")) 

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

                # resize image so it's same as test images
                center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
                half_crop = 512//2
                img = np.array(img[center_y -half_crop : center_y + half_crop, center_x - half_crop : center_x + half_crop])

                # store it in a dict for later (greatly accelerates subsequent epochs)
                images_cached[item] = img
                #size = get_size(images_cached)
                #print(size / (1024. * 1024. * 1024.))

            else:
                img = images_cached[item]

            # to save time use sub_batch_size crops from same image on the same batch
            # does not really matter images have been cached on images_cached dict.
            sub_batch_size = args.sub_batch_size
            assert batch_size % sub_batch_size == 0

            manipulated = 0.
            if not inference and np.random.rand() < 0.5:
                img = random_manipulation(img)
                manipulated = 1.

            if args.kernel_filter:
                # see slide 13
                # http://www.lirmm.fr/~chaumont/publications/WIFS-2016_TUAMA_COMBY_CHAUMONT_Camera_Model_Identification_With_CNN_slides.pdf
                kernel_filter = 1/12. * np.array([\
                    [-1,  2,  -2,  2, -1],  \
                    [ 2, -6,   8, -6,  2],  \
                    [-2,  8, -12,  8, -2],  \
                    [ 2, -6,   8, -6,  2],  \
                    [-1,  2,  -2,  2, -1]]) 

                img = cv2.filter2D(img.astype(np.float32),-1,kernel_filter)
                # kernel filter already puts mean ~0 and roughly scales between [-1..1]
                # no need to preprocess_input further
                
            else:
                img = preprocess_image(img)

            for sub_batch in range(sub_batch_size):

                # right now just take corners of image as crops
                sx= 0 #random.randint(0, 1)
                sy= 0 #random.randint(0, 1)

                _sx = img.shape[1] - CROP_SIZE if sx == 1 else 0
                _sy = img.shape[0] - CROP_SIZE if sy == 1 else 0

                _img = img[_sy:_sy+CROP_SIZE, _sx:_sx+CROP_SIZE]

                X[batch_idx+sub_batch] = _img
                O[batch_idx+sub_batch] = manipulated - 0.5 # between -0.5 an 0.5
                y[batch_idx+sub_batch] = class_idx

            if batch_idx == 0 and False: #remove False if you want to see images on stdout (requires iterm2)
                show_image(X[batch_idx])

            batch_idx += sub_batch_size

            if batch_idx == batch_size:

                if not inference:
                    # commented b/c converges slower... why???

                    #I_O_zipped = list(zip(X, O, y))
                    #random.shuffle(I_O_zipped)
                    #X[:], O[:], y[:] = zip(*I_O_zipped)
                    yield([X, O], [y])
                
                else:   
                    yield([X, O])

                batch_idx = 0

        if not os.path.isfile(args.preprocessed_input_path) and training:
            pickle.dump(images_cached, open(args.preprocessed_input_path, 'wb'))

# attempt to use Sequence instead of generator to allow multiprocessing 
# but still it's substantially slower than generator with cached images
class CameraImagesSequence(Sequence):

    def __init__(self, items, batch_size, training=True, inference=False):
        self.X_path = items
        self.y      = [CLASSES.index(item.split('/')[-2]) for item in items]

        self.batch_size = batch_size
        self.training   = training
        self.inference  = inference

    def __len__(self):
        return math.ceil(len(self.X_path) / self.batch_size)

    def __getitem__(self, idx):
        batch_X_path = self.X_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y      = self.y     [idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros((self.batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

        # O holds normalized float of where the crop is in the image (local relative position)
        O = np.zeros((self.batch_size, 2), dtype=np.float32)
        y = np.zeros((self.batch_size),    dtype=np.int64)

        for it, (item, class_idx) in enumerate(zip(batch_X_path, batch_y)):
            img = load_img(item)

            # resize image so it's same as test images
            img = skimage.transform.resize(img, (512, 512), mode='reflect')

            if args.kernel_filter:
                # see slide 13
                # http://www.lirmm.fr/~chaumont/publications/WIFS-2016_TUAMA_COMBY_CHAUMONT_Camera_Model_Identification_With_CNN_slides.pdf
                kernel_filter = 1/12. * np.array([\
                    [-1,  2,  -2,  2, -1],  \
                    [ 2, -6,   8, -6,  2],  \
                    [-2,  8, -12,  8, -2],  \
                    [ 2, -6,   8, -6,  2],  \
                    [-1,  2,  -2,  2, -1]]) 

                img = cv2.filter2D(img.astype(np.float32),-1,kernel_filter)
                # kernel filter already puts mean ~0 and roughly scales between [-1..1]
                # no need to preprocess_input further
            else:
                # find `preprocess_input` function specific to the classifier
                classifier_to_module = { 
                    'NASNetLarge'       : 'nasnet',
                    'NASNetMobile'      : 'nasnet',
                    'DenseNet121'       : 'densenet',
                    'DenseNet161'       : 'densenet',
                    'DenseNet201'       : 'densenet',
                    'InceptionResNetV2' : 'inception_resnet_v2',
                    'InceptionV3'       : 'inception_v3',
                    'MobileNet'         : 'mobilenet',
                    'ResNet50'          : 'resnet50',
                    'VGG16'             : 'vgg16',
                    'VGG19'             : 'vgg19',
                    'Xception'          : 'xception',
                }

                classifier_module_name = classifier_to_module[args.classifier]
                preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
                img = preprocess_input_function(img.astype(np.float32))

            # at this point we now have img
            sx= random.randint(0, 1)
            sy= random.randint(0, 1)

            _sx = img.shape[1] - CROP_SIZE if sx == 1 else 0
            _sy = img.shape[0] - CROP_SIZE if sy == 1 else 0

            _img = img[_sy:_sy+CROP_SIZE, _sx:_sx+CROP_SIZE]

            X[it] = _img
            O[it] = np.float32([sx, sy]) - np.float32([0.5,0.5])
            y[it] = class_idx

        return [X, O], y

    # instead could have used shuffle=True on .fit_generator(...) https://keras.io/models/model/
    def on_epoch_end(self):
            """Method called at the end of every epoch.
            """
            print('\nEpoch end' )
            if self.training:
                print("Shuffling")
                I_O_zipped = list(zip(self.X_path, self.y))
                random.shuffle(I_O_zipped)
                self.X_path, self.y = zip(*I_O_zipped)

# Custom layer used to do filtering on the GPU
# Not sure if it yields correct results so not using right now.
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
                    shape=(5,5,3,3), # need to check this!
                    initializer=self.kernel_initializer,
                    trainable=False)

                super(KernelFilter, self).build(input_shape)  # Be sure to call this somewhere!

        def call(self, x):

                print(x.shape)
                print(self.kernel)

                return K.conv2d(x, self.kernel, padding='same', data_format='channels_last')

if args.model:
    print("Loading model " + args.model)

    model = load_model(args.model, compile=False)
    match = re.search(r'([a-z]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(2)) + 1

else:
    last_epoch = 0

    input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3))
    manipulated = Input(shape=(1,))
    image_filtered = input_image # not using this -> KernelFilter(input_shape=(CROP_SIZE, CROP_SIZE, 3))(input_image)

    classifier = globals()[args.classifier]

    classifier_model = classifier(
        include_top=False, 
        weights = 'imagenet' if args.use_imagenet_weights else None,
        input_shape=(CROP_SIZE, CROP_SIZE, 3), 
        pooling=args.pooling if args.pooling != 'none' else None)

    x = classifier_model(image_filtered)
    if args.pooling == 'none':
        x = Flatten()(x)
    x = Dropout(args.dropout)(x)
    x = concatenate([x,manipulated])
    prediction = Dense(N_CLASSES, activation ="softmax", name="predictions")(x)

    model = Model(inputs=(input_image, manipulated), outputs=prediction)

    model.summary()

    model_name = args.classifier + ('_kf' if args.kernel_filter else '') + '_do' + str(args.dropout)

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

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')

    model.fit_generator(
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = int(math.ceil(len(ids_val) // args.batch_size)),
            epochs = args.max_epoch,
            callbacks = [save_checkpoint, reduce_lr],
            initial_epoch = last_epoch)
            #use_multiprocessing=True, 
            #workers=31,
            #max_queue_size=16)