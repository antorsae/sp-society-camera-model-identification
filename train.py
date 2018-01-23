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
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import to_categorical, Sequence
from keras.initializers import Constant
from keras.applications import *

from multi_gpu_keras import multi_gpu_model
from keras import backend as K
from keras.engine.topology import Layer
import skimage
import pickle
from iterm import show_image
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import itertools
import re
import os
import sys
from tqdm import tqdm
import jpeg4py as jpeg
from scipy import signal
import cv2
import math
import csv
from sklearn.utils import class_weight

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=100, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-pp','--preprocessed_input_path', type=str, default='images_preprocessed', help='Path to preprocessed images')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')

args = parser.parse_args()

args.preprocessed_input_path += '-' + str(args.crop_size) + ('-x.pkl' if args.extra_dataset else '.pkl')

TRAIN_FOLDER       = 'train'
EXTRA_TRAIN_FOLDER = 'flickr_images'
EXTRA_VAL_FOLDER   = 'val_images'
TEST_FOLDER        = 'test'
MODEL_FOLDER       = 'models'

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

EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]

MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

N_CLASSES = len(CLASSES)
#load_img  = lambda img_path: jpeg.JPEG(img_path).decode()
load_img  = lambda img_path: np.array(Image.open(img_path))

def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    return im_decoded

def preprocess_image(img):
    
    if args.kernel_filter:
        # see slide 13
        # http://www.lirmm.fr/~chaumont/publications/WIFS-2016_TUAMA_COMBY_CHAUMONT_Camera_Model_Identification_With_CNN_slides.pdf
        kernel_filter = 1/12. * np.array([\
            [-1,  2,  -2,  2, -1],  \
            [ 2, -6,   8, -6,  2],  \
            [-2,  8, -12,  8, -2],  \
            [ 2, -6,   8, -6,  2],  \
            [-1,  2,  -2,  2, -1]]) 

        return cv2.filter2D(img.astype(np.float32),-1,kernel_filter)
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

        if args.classifier in classifier_to_module:
            classifier_module_name = classifier_to_module[args.classifier]
        else:
            classifier_module_name = 'xception'

        preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
        return preprocess_input_function(img.astype(np.float32))

def get_crop(img, crop_size):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='reflect')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]

def get_class(class_name):
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        assert False

    assert class_idx in range(N_CLASSES)
    return class_idx

def gen(items, batch_size, training=True, inference=False):

    images_cached = { }

    validation = not training and not inference
    training   = training and not inference

    valid_batch_factor = 2 if validation else 1

    # X holds image crops
    X = np.zeros((batch_size * valid_batch_factor, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    # O whether the image has been manipulated (1.) or not (0.)
    O = np.zeros((batch_size * valid_batch_factor, 1), dtype=np.float32)

    if not inference:
        y = np.zeros((batch_size * valid_batch_factor), dtype=np.int64)

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

                class_idx = get_class(class_name)

            if item not in images_cached:

                img = load_img(item)
                if img.ndim != 3:
                    # some images may not be downloaded correclty and are B/W, skip those
                    #print(item, img.shape)
                    continue
                img = np.array(get_crop(img, CROP_SIZE * 2)) # * 2 bc many need to scale by 0.5x and still get a 512px crop
                # store it in a dict for later (greatly accelerates subsequent epochs)
                images_cached[item] = img

            img =  np.array(images_cached[item])

            if validation:
                unalt_img = np.array(img)

            manipulated = 0.
            if (np.random.rand() < 0.5) and training:
                img = random_manipulation(img)
                manipulated = 1.

            img = get_crop(img, CROP_SIZE)
            img = preprocess_image(img)
            X[batch_idx] = img
            O[batch_idx] = manipulated

            if validation:
                manip_img = random_manipulation(unalt_img)
                manip_img = get_crop(manip_img, CROP_SIZE)
                manip_img = preprocess_image(manip_img)
                X[batch_idx + batch_size] = manip_img
                O[batch_idx + batch_size] = 1. # manipulated 

            if not inference:
                y[batch_idx] = class_idx

                if validation:
                    y[batch_idx + batch_size] = class_idx

            if batch_idx == 0 and False: #remove False if you want to see images on stdout (requires iterm2)
                show_image(X[batch_idx])

            batch_idx += 1

            if batch_idx == batch_size:

                if not inference:

                    yield([X, O], [y])
                
                else:   
                    yield([X, O])

                batch_idx = 0

        if not os.path.isfile(args.preprocessed_input_path) and training:
            pickle.dump(images_cached, open(args.preprocessed_input_path, 'wb'))

def SmallNet(include_top, weights, input_shape, pooling):
    img_input = Input(shape=input_shape)
    x = Conv2D( 3, (7, 7), strides=(1,1), padding='valid', name='filtering')(img_input)

    x = Conv2D(64, (3, 3), strides=(2,2), padding='valid', name='conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(2,2), padding='valid', name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), strides=(1,1), padding='valid', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x)

    model = Model(img_input, x, name='smallnet')

    return model

# MAIN

if args.model:
    print("Loading model " + args.model)

    model = load_model(args.model, compile=False)
    match = re.search(r'([A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(2))
else:
    last_epoch = 0

    input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3))
    manipulated = Input(shape=(1,))

    classifier = globals()[args.classifier]

    classifier_model = classifier(
        include_top=False, 
        weights = 'imagenet' if args.use_imagenet_weights else None,
        input_shape=(CROP_SIZE, CROP_SIZE, 3), 
        pooling=args.pooling if args.pooling != 'none' else None)

    #x = Conv2D(3, (7, 7), strides=(1,1), use_bias=False, padding='valid', name='filtering')(image_filtered)
    x = input_image
    x = classifier_model(x)
    if args.pooling == 'none':
        x = Flatten()(x)
    x = concatenate([x, manipulated])
    x = Dense(256*2, activation='relu')(x)
    x = Dropout(args.dropout)(x)
    x = Dense(128*2,  activation='relu')(x)
    x = Dropout(args.dropout)(x)
    prediction = Dense(N_CLASSES, activation ="softmax", name="predictions")(x)

    model = Model(inputs=(input_image, manipulated), outputs=prediction)
    model_name = args.classifier + ('_kf' if args.kernel_filter else '') + '_do' + str(args.dropout) + '_' + args.pooling

model.summary()
model = multi_gpu_model(model, gpus=args.gpus)

if not (args.test or args.test_train):
    # TRAINING

    ids = glob.glob(join(TRAIN_FOLDER,'*/*.jpg'))
    ids.sort()

    if not args.extra_dataset:
        ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)
    else:
        ids_train = ids
        ids_val   = [ ]

        extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
        extra_train_ids.sort()
        ids_train.extend(extra_train_ids)

        extra_val_ids = glob.glob(join(EXTRA_VAL_FOLDER,'*/*.jpg'))
        extra_val_ids.sort()
        ids_val.extend(extra_val_ids)

    classes = [get_class(idx.split('/')[-2]) for idx in ids_train]

    classes_count = np.bincount(classes)
    for class_name, class_count in zip(CLASSES, classes_count):
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))

    class_weight = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

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

        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')

        model.fit_generator(
                generator        = gen(ids_train, args.batch_size),
                steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
                validation_data  = gen(ids_val, args.batch_size, training = False),
                validation_steps = int(math.ceil(len(ids_val) // args.batch_size)),
                epochs = args.max_epoch,
                callbacks = [save_checkpoint, reduce_lr],
                initial_epoch = last_epoch,
                class_weight=class_weight)

else:
    # TEST
    if args.test:
        ids = glob.glob(join(TEST_FOLDER,'*.tif'))
    elif args.test_train:
        ids = glob.glob(join(TRAIN_FOLDER,'*/*.jpg'))
    else:
        assert False

    ids.sort()
    
    from conditional import conditional

    with conditional(args.test, open('submission.csv', 'w')) as csvfile:

        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname','camera'])
        else:
            correct_predictions = 0

        for i, idx in enumerate(tqdm(ids)):

            img = np.array(Image.open(idx))

            img = get_crop(img, CROP_SIZE)
            img = np.expand_dims(preprocess_image(img), axis=0)

            manipulated = np.float32([[1. if idx.find('manip') != -1 else 0.]])
            # TODO: TTA when not manipulated -> perform argmax on all manipulations

            prediction = model.predict_on_batch([img,manipulated])
            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = get_class(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:

                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
        if args.test_train:
            print("Accuracy: " + str(correct_predictions / i))
                