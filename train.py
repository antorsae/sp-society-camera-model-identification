# IEEE's Signal Processing Society - Camera Model Identification 
# https://www.kaggle.com/c/sp-society-camera-model-identification
#
# (C) 2018 Andres Torrubia, licensed under GNU General Public License v3.0 
# See license.txt

import argparse
import glob
import numpy as np
import pandas as pd
import random
from os.path import join

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, SeparableConv2D
from keras.utils import to_categorical
from keras.applications import *
from keras import backend as K
from keras.engine.topology import Layer

from keras_squeeze_excite_network import * 

from multi_gpu_keras import multi_gpu_model

import skimage
from iterm import show_image

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import copy
import itertools
import re
import os
import sys
import jpeg4py as jpeg
from scipy import signal
import cv2
import math
import csv
from multiprocessing import Pool, cpu_count

from functools import partial
from itertools import  islice
from conditional import conditional

from clr_callback import *

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

parser = argparse.ArgumentParser()
# general
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-clr', '--cyclic_learning_rate',action='store_true', help='Use cyclic learning rate https://arxiv.org/abs/1506.01186')
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to use in training -o adam|sgd|adadelta')
parser.add_argument('--amsgrad', action='store_true', help='Apply the AMSGrad variant of adam|adadelta from the paper "On the Convergence of Adam and Beyond".')

# architecture/model
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for first FC layer')
parser.add_argument('-dol', '--dropout-last', type=float, default=0.1, help='Dropout rate for last FC layer')
parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
parser.add_argument('-fc', '--fully-connected-layers', nargs='+', type=int, default=[512,256], help='Specify FC layers after classifier, e.g. -fc 1024 512 256')
parser.add_argument('-bn', '--batch-normalization', action='store_true', help='Use batch normalization in FC layers')
parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
parser.add_argument('-rp', '--reduce-pooling', type=int, default=None, help='If using pooling none add conv layers to reduce features, e.g. -rp 128')

# training regime
parser.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
parser.add_argument('-cc', '--center-crops', nargs='*', type=int, default=[], help='Train on center crops only (not random crops) for the selected classes e.g. -cc 1 6 or all -cc -1')
parser.add_argument('-nf', '--no-flips', action='store_true', help='Dont use orientation flips for augmentation')
parser.add_argument('-fcm', '--freeze-classifier', action='store_true', help='Freeze classifier weights (useful to fine-tune FC layers)')

# dataset (training)
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')

# test
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-tcs', '--test-crop-supersampling', default=1, type=int, help='Factor of extra crops to sample during test, especially useful when crop size is less than 512, e.g. -tcs 4')
parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')
parser.add_argument('-e', '--ensembling', type=str, default='arithmetic', help='Type of ensembling: arithmetic|geometric for TTA')


args = parser.parse_args()

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

N_CLASSES = len(CLASSES)

if args.center_crops==[-1]:
    args.center_crops = range(N_CLASSES)

TRAIN_FOLDER       = 'train'
EXTRA_TRAIN_FOLDER = 'flickr_images'
EXTRA_VAL_FOLDER   = 'val_images'
TEST_FOLDER        = 'test'
MODEL_FOLDER       = 'models'

CROP_SIZE = args.crop_size

# See motivsation for this:
# https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/48680
# and http://sylvana.net/jpegcrop/exif_orientation.html
#
# These are resolutions, and probabilities are from the original training set;
#
#       [size_y, size_x, probability]. Sum of probabilites for each model should add up to 1. 
#
RESOLUTIONS = {                # samples*2  model      samples   EXIF orientation
#
    0: [[2688,1520, 508/550],  # 508        htc_m7         275   Orientation: TopLeft
        [1520,2688,  42/550]], #  42
    1: [[2448,3264,       1]], # 550        iphone_6       273   Orientation: RightTop
                               #                             2   Orientation: TopLeft    
    2: [[2432,4320, 472/550],  # 472        moto_maxx      275   Orientation: Undefined
        [4320,2432,  78/550]], #  78 
    3: [[3120,4160, 468/550],  # 468        moto_x         275   Orientation: Undefined
        [4160,3120,  82/550]], #  82
    4: [[2322,4128,       1]], # 550        samsung_s4     275   Orientation: RightTop
    5: [[2448,3264,       1]], # 550        iphone 4s      275   Orientation: RightTop
    6: [[4032,3024, 544/550],  # 544        nexus_5x       275   Orientation: Undefined
        [3024,4032,   6/550]], #   6 
    7: [[780, 1040,   2/550],  #   2        nexus_6          1   Orientation: LeftBottom
        [4130,3088,   2/550],  #   2                       274   Orientation: TopLeft
        [4160,3088,  30/550],  #  30
        [4160,3120, 256/550],  # 256
        [3088,4160,  18/550],  #  18
        [3120,4160, 242/550]], # 242
    8: [[2322,4128,       1]], # 550        samsung_note3  196   Orientation: RightTop
                               #                            79   Orientation: TopLeft
    9: [[4000,6000,       1]], # 550        sony_nex7       35   Orientation: LeftBottom
                               #                             3   Orientation: RightTop
                               #                           237   Orientation: TopLeft
}

ORIENTATION_FLIP_ALLOWED = [
    True,  # htc_m7
    False, # iphone_6
    True,  # moto_maxx
    True,  # moto_x
    False, # samsung_s4
    False, # iphone 4s   
    True,  # nexus_5x
    False, # Motorola-Nexus-6
    False, # samsung_note3
    False  # sony_nex7
]


#for class_id,resolutions in RESOLUTIONS.copy().items():
#    if ORIENTATION_FLIP_ALLOWED[class_id]:
##        resolutions.extend([resolution[::-1] for resolution in resolutions])
#    RESOLUTIONS[class_id] = resolutions

MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
load_img           = lambda img_path: np.array(Image.open(img_path))

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
    else:
        assert False
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
            'DenseNet40'        : 'densenet',
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

            'SEDenseNetImageNet121' : 'se_densenet',
            'SEDenseNetImageNet161' : 'se_densenet',
            'SEDenseNetImageNet169' : 'se_densenet',
            'SEDenseNetImageNet264' : 'se_densenet',
            'SEInceptionResNetV2'   : 'se_inception_resnet_v2',
            'SEMobileNet'           : 'se_mobilenets',
            'SEResNet50'            : 'se_resnet',
            'SEResNet101'           : 'se_resnet',
            'SEResNet154'           : 'se_resnet',
            'SEInceptionV3'         : 'se_inception_v3',
            'SEResNext'             : 'se_resnet',
            'SEResNextImageNet'     : 'se_resnet',

        }

        if args.classifier in classifier_to_module:
            classifier_module_name = classifier_to_module[args.classifier]
        else:
            classifier_module_name = 'xception'

        preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
        return preprocess_input_function(img.astype(np.float32))

def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

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

def process_item(item, training, transforms=[[]]):

    class_name = item.split('/')[-2]
    class_idx = get_class(class_name)

    validation = not training 

    img = load_img_fast_jpg(item)

    shape = list(img.shape[:2])

    # discard images that do not have right resolution
    if shape not in [resolution[:2] for resolution in RESOLUTIONS[class_idx]]:
        return None

    # some images may not be downloaded correctly and are B/W, discard those
    if img.ndim != 3:
        return None

    if len(transforms) == 1:
        _img = img
    else:
        _img = np.copy(img)

        img_s         = [ ]
        manipulated_s = [ ]
        class_idx_s   = [ ]

    for transform in transforms:

        force_manipulation = 'manipulation' in transform

        if args.no_flips and (('orientation' in transform) and (ORIENTATION_FLIP_ALLOWED[class_idx] is False)):
            continue

        force_orientation  = not args.no_flips and ('orientation'  in transform) and ORIENTATION_FLIP_ALLOWED[class_idx] 

        # some images are landscape, others are portrait, so augment training by randomly changing orientation
        if not args.no_flips and ((((np.random.rand() < 0.5) and training and ORIENTATION_FLIP_ALLOWED[class_idx])) or force_orientation):
            img = np.rot90(_img, 1, (0,1))
            # is it rot90(..3..), rot90(..1..) or both? 
            # for phones with landscape mode pics could be taken upside down too, although less likely
            # most of the test images that are flipped are 1
            # however,eg. img_4d7be4c_unalt looks 3
            # and img_4df3673_manip img_6a31fd7_unalt looks 2!
        else:
            img = _img

        # TESTING
        # Correct orientstion for Nexus 5
        if (class_idx == 6):
            canonical_resolution = tuple(RESOLUTIONS[class_idx][0][:2])
            if img.shape[:2] != canonical_resolution:
                img = np.rot90(img, 1, (0,1))
                assert img.shape[:2] == canonical_resolution
 

        img = get_crop(img, CROP_SIZE * 2, random_crop=(class_idx not in args.center_crops) if training else False) 
        # * 2 bc may need to scale by 0.5x and still get a 512px crop

        if args.verbose:
            print("om: ", img.shape, item)

        manipulated = 0.
        if ((np.random.rand() < 0.5) and training) or force_manipulation:
            img = random_manipulation(img)
            manipulated = 1.
            if args.verbose:
                print("am: ", img.shape, item)

        img = get_crop(img, CROP_SIZE, random_crop=(class_idx not in args.center_crops) if training else False)
        if args.verbose:
            print("ac: ", img.shape, item)

        img = preprocess_image(img)
        if args.verbose:
            print("ap: ", img.shape, item)

        if len(transforms) > 1:
            img_s.append(img)    
            manipulated_s.append(manipulated)
            class_idx_s.append(class_idx)

    if len(transforms) == 1:
        return img, manipulated, class_idx
    else:
        return img_s, manipulated_s, class_idx_s

VALIDATION_TRANSFORMS = [ [], ['orientation'], ['manipulation'], ['orientation','manipulation']]

def gen(items, batch_size, training=True):

    validation = not training 

    # during validation we store the unaltered images on batch_idx and a manip one on batch_idx + batch_size, hence the 2
    valid_batch_factor = 1 # TODO: augment validation

    # X holds image crops
    X = np.empty((batch_size * valid_batch_factor, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    # O whether the image has been manipulated (1.) or not (0.)
    O = np.empty((batch_size * valid_batch_factor, 1), dtype=np.float32)

    # class index
    y = np.empty((batch_size * valid_batch_factor), dtype=np.int64)
    
    p = Pool(cpu_count()-2)

    transforms = VALIDATION_TRANSFORMS if validation else [[]]

    bad_items = set()
    while True:

        if training:
            random.shuffle(items)

        process_item_func  = partial(process_item, training=training, transforms=transforms)

        batch_idx = 0
        iter_items = iter(items)
        for item_batch in iter(lambda:list(islice(iter_items, batch_size)), []):

            batch_results = p.map(process_item_func, item_batch)
            for batch_result, item in zip(batch_results, item_batch):

                if batch_result is not None:
                    if len(transforms) == 1:
                        X[batch_idx], O[batch_idx], y[batch_idx] = batch_result
                        batch_idx += 1
                    else:
                        for _X,_O,_y in zip(*batch_result):
                            X[batch_idx], O[batch_idx], y[batch_idx] = _X,_O,_y
                            batch_idx += 1
                            if batch_idx == batch_size:
                                yield([X, O], [y])
                                batch_idx = 0
                else: # if batch result is None
                    bad_items.add(item)

                if batch_idx == batch_size:
                    yield([X, O], [y])
                    batch_idx = 0

        if len(bad_items) > 0:
            print("\nRejected {} items: {}".format('trainining' if training else 'validation', len(bad_items)))
            print(bad_items)

def SmallNet(include_top, weights, input_shape, pooling):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2,2), padding='valid', name='conv1')(img_input)
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

# see https://arxiv.org/pdf/1703.04856.pdf
def CaCNN(include_top, weights, input_shape, pooling):

    img_input = Input(shape=input_shape)

    def CaCNNBlock(x, preffix=''):
        x = Conv2D(  8, (3, 3), strides=(1,1), padding='valid', name=preffix+'conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D( 16, (3, 3), strides=(1,1), padding='valid', name=preffix+'conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D( 32, (3, 3), strides=(1,1), padding='valid', name=preffix+'conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D( 64, (3, 3), strides=(1,1), padding='valid', name=preffix+'conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), strides=(1,1), padding='valid', name=preffix+'conv5')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D(name=preffix+'pooling')(x)  
        
        return x

    x = img_input

    x1 = Conv2D(3, (3, 3), use_bias=False, padding='valid', name='filter1')(x)
    x1 = BatchNormalization()(x1)
    x1 = CaCNNBlock(x1, preffix='block1')

    x2 = Conv2D(3, (5, 5), use_bias=False, padding='valid', name='filter2')(x)
    x2 = BatchNormalization()(x2)
    x2 = CaCNNBlock(x2, preffix='block2')

    x3 = Conv2D(3, (7, 7), use_bias=False, padding='valid', name='filter3')(x)
    x3 = BatchNormalization()(x3)
    x3 = CaCNNBlock(x3, preffix='block3')

    x = concatenate([x1,x2,x3])

    model = Model(img_input, x, name='cacnn')

    model.summary()

    return model

# MAIN
if args.model:
    print("Loading model " + args.model)

    model = load_model(args.model, compile=False)
    # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
    match = re.search(r'(([a-zA-Z\d]+)_cs[,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    args.classifier = match.group(2)
    CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]
    print("Overriding classifier: {} and crop size: {}".format(args.classifier, args.crop_size))
    last_epoch = int(match.group(3))
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

    x = input_image
    if args.learn_kernel_filter:
        x = Conv2D(3, (7, 7), strides=(1,1), use_bias=False, padding='valid', name='filtering')(x)
    x = classifier_model(x)

    if args.reduce_pooling and x.shape.ndims == 4:

        pool_features = int(x.shape[3])

        for it in range(int(math.log2(pool_features/args.reduce_pooling))):

            pool_features //= 2
            x = Conv2D(pool_features, (3, 3), padding='same', use_bias=False, name='reduce_pooling{}'.format(it))(x)
            x = BatchNormalization(name='bn_reduce_pooling{}'.format(it))(x)
            x = Activation('relu', name='relu_reduce_pooling{}'.format(it))(x)
        
    x = Reshape((-1,))(x)
    if args.dropout_classifier != 0.:
        x = Dropout(args.dropout_classifier, name='dropout_classifier')(x)
    x = concatenate([x, manipulated])
    if not args.no_fcs:
        dropouts = np.linspace( args.dropout,  args.dropout_last, len(args.fully_connected_layers))
        for i, (fc_layer, dropout) in enumerate(zip(args.fully_connected_layers, dropouts)):
            if args.batch_normalization:
                x = Dense(fc_layer,    name='fc{}'.format(i))(x)
                x = BatchNormalization(name='bn{}'.format(i))(x)
                x = Activation('relu', name='relu{}'.format(i))(x)
            else:
                x = Dense(fc_layer, activation='relu', name='fc{}'.format(i))(x)
            if dropout != 0:
                x = Dropout(dropout,                   name='dropout_fc{}_{:04.2f}'.format(i, dropout))(x)
    prediction = Dense(N_CLASSES, activation ="softmax", name="predictions")(x)

    model = Model(inputs=(input_image, manipulated), outputs=prediction)
    model_name = args.classifier + \
        '_cs{}'.format(args.crop_size) + \
        ('_fc{}'.format(','.join([str(fc) for fc in args.fully_connected_layers])) if not args.no_fcs else '_nofc') + \
        ('_bn' if args.batch_normalization else '') + \
        ('_kf' if args.kernel_filter else '') + \
        ('_lkf' if args.learn_kernel_filter else '') + \
        '_doc' + str(args.dropout_classifier) + \
        '_do'  + str(args.dropout) + \
        '_dol' + str(args.dropout_last) + \
        '_' + args.pooling + \
        ('_x' if args.extra_dataset else '') + \
        ('_cc{}'.format(','.join([str(c) for c in args.center_crops])) if args.center_crops else '') + \
        ('_nf' if args.no_flips else '') 

    print("Model name: " + model_name)

    if args.weights:
            model.load_weights(args.weights, by_name=True, skip_mismatch=True)
            match = re.search(r'([,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.weights)
            last_epoch = int(match.group(2))

def print_distribution(ids, classes=None, prediction_probabilities=None):
    if classes is None:
        classes = [get_class(idx.split('/')[-2]) for idx in ids]
    classes=np.array(classes)
    classes_count = np.bincount(classes)
    threshold = 0.7
    poor_prediction_probabilities = 0
    for class_idx, (class_name, class_count) in enumerate(zip(CLASSES, classes_count)):
        if prediction_probabilities is not None:
            prediction_probabilities_this_class = prediction_probabilities[classes == class_idx, class_idx]
            poor_prediction_probabilities_this_class = (prediction_probabilities_this_class < threshold ).sum()
            poor_prediction_probabilities += poor_prediction_probabilities_this_class
            poor_prediction_probabilities_this_class /= prediction_probabilities_this_class.size
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)) + \
            (' Poor predictions: {:04.1f}%'.format(100 * poor_prediction_probabilities_this_class) if prediction_probabilities is not None else ''))
    if prediction_probabilities is not None:
        print("                                Total poor predictions: {:04.1f}% (threshold = {:03.1f})".format( \
            100. * poor_prediction_probabilities / classes.size, threshold))

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

        extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
        low_quality     = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n').split(' ')[0]) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'low_quality'))]
        bad_resolution  = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n').split(' ')[0]) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'bad_resolution'))]
        reject_ids = low_quality + bad_resolution
        extra_train_ids = [idx for idx in extra_train_ids if idx not in reject_ids]
        extra_train_ids.sort()
        ids_train.extend(extra_train_ids)
        random.shuffle(ids_train)

        extra_val_ids = glob.glob(join(EXTRA_VAL_FOLDER,'*/*.jpg'))
        extra_val_ids.sort()
        ids_val.extend(extra_val_ids)

        classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]
        classes_val_count = np.bincount(classes_val)
        max_classes_val_count = max(classes_val_count)

        # Balance validation dataset by filling up classes with less items from training set (and removing those from there)
        for class_idx in range(N_CLASSES):
            idx_to_transfer = [idx for idx in ids_train \
                if get_class(idx.split('/')[-2]) == class_idx][:max_classes_val_count-classes_val_count[class_idx]]

            ids_train = list(set(ids_train).difference(set(idx_to_transfer)))

            ids_val.extend(idx_to_transfer)

        random.shuffle(ids_val)

    print("Training set distribution:")
    print_distribution(ids_train)

    print("Validation set distribution:")
    print_distribution(ids_val)
    print("Total training items: {}\nTotal validation items: {}".format(len(ids_train), len(ids_val)))

    classes_train = [get_class(idx.split('/')[-2]) for idx in ids_train]
    class_weight = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

    if args.optimizer == 'adam':
        opt = Adam(lr=args.learning_rate, amsgrad=args.amsgrad)
    elif args.optimizer == 'sgd':
        opt = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adadelta':
        opt = Adadelta(lr=args.learning_rate, amsgrad=args.amsgrad)
    else:
        assert False

    # TODO: implement this correctly.
    def weighted_loss(weights):
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
        return loss

    if args.freeze_classifier:
        for layer in model.layers:
            if isinstance(layer, Model):
                print("Freezing weights for classifier {}".format(layer.name))
                for classifier_layer in layer.layers:
                    classifier_layer.trainable = False

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    metric  = "-val_acc{val_acc:.6f}"
    monitor = 'val_acc'

    save_checkpoint = ModelCheckpoint(
            join(MODEL_FOLDER, model_name+"-epoch{epoch:03d}"+metric+".hdf5"),
            monitor=monitor,
            verbose=0,  save_best_only=True, save_weights_only=False, mode='max', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-9, epsilon = 0.00001, verbose=1, mode='max')

    orientation_flip_augmentation_factor = (sum(ORIENTATION_FLIP_ALLOWED) / len(ORIENTATION_FLIP_ALLOWED)) if not args.no_flips else 0.5
    
    clr = CyclicLR(base_lr=args.learning_rate, max_lr=args.learning_rate*10,
                        step_size=int(math.ceil(len(ids_train)  // args.batch_size)) * 4, mode='exp_range',
                        gamma=0.99994)

    callbacks = [save_checkpoint]

    if args.cyclic_learning_rate:
        callbacks.append(clr)
    else:
        callbacks.append(reduce_lr)
    
    model.fit_generator(
            generator        = gen(ids_train, args.batch_size),
            steps_per_epoch  = int(math.ceil(len(ids_train)  // args.batch_size)),
            validation_data  = gen(ids_val, args.batch_size, training = False),
            validation_steps = int(len(VALIDATION_TRANSFORMS) * orientation_flip_augmentation_factor  \
                                   * math.ceil(len(ids_val) // args.batch_size)),
            epochs = args.max_epoch,
            callbacks = callbacks,
            initial_epoch = last_epoch,
            max_queue_size = 10,
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

    match = re.search(r'([^/]*)\.hdf5', args.model)
    model_name = match.group(1) + ('_tta_' + args.ensembling if args.tta else '')
    csv_name   = 'submission_' + model_name + '.csv'
    with conditional(args.test, open(csv_name, 'w')) as csvfile:

        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname','camera'])
            classes = []
            prediction_probabilities = []
        else:
            correct_predictions = 0

        for i, idx in enumerate(tqdm(ids)):

            img = np.array(Image.open(idx))

            if args.test_train:
                img = get_crop(img, 512*2, random_crop=False)

            original_img = img

            original_manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])


            if args.test and args.tta:
                transforms = [[], ['orientation']]
            elif args.test_train:
                transforms = [[], ['orientation'], ['manipulation'], ['manipulation', 'orientation']]
            else:
                transforms = [[]]

            ssx = int(img.shape[1] / CROP_SIZE * args.test_crop_supersampling)
            ssy = int(img.shape[0] / CROP_SIZE * args.test_crop_supersampling)

            img_batch         = np.zeros((len(transforms)* ssx * ssy, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
            manipulated_batch = np.zeros((len(transforms)* ssx * ssy, 1),  dtype=np.float32)

            i = 0
            for transform in transforms:
                img = np.copy(original_img)
                manipulated = np.copy(original_manipulated)

                if 'orientation' in transform:
                    img = np.rot90(img, 1, (0,1))
                if 'manipulation' in transform and not original_manipulated:
                    img = random_manipulation(img)
                    manipulated = np.float32([1.])

                if args.test_train:
                    img = get_crop(img, 512, random_crop=False)

                sx = img.shape[1] / CROP_SIZE
                sy = img.shape[0] / CROP_SIZE

                for x in np.linspace(0, img.shape[1] - CROP_SIZE, args.test_crop_supersampling * sx, dtype=np.int64):
                    for y in np.linspace(0, img.shape[0] - CROP_SIZE, args.test_crop_supersampling * sy, dtype=np.int64):
                        _img = np.copy(img[y:y+CROP_SIZE, x:x+CROP_SIZE])
                        img_batch[i]         = preprocess_image(_img)
                        manipulated_batch[i] = manipulated
                        i += 1

            prediction = model.predict_on_batch([img_batch,manipulated_batch])
            if prediction.shape[0] != 1: # TTA
                if args.ensembling == 'geometric':
                    predictions = np.log(prediction + K.epsilon()) # avoid numerical instability log(0)
                prediction = np.mean(prediction, axis=0)

            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = get_class(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:
                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
                classes.append(prediction_class_idx)
                prediction_probabilities.append(prediction)

        if args.test_train:
            print("Accuracy: " + str(correct_predictions / (len(transforms) * i)))

        if args.test:
            prediction_probabilities = np.squeeze(np.array(prediction_probabilities))
            print("Test set predictions distribution:")
            print_distribution(None, classes=classes, prediction_probabilities=prediction_probabilities)
            print("Predictions as per old-school model inference:")
            print("kg submit {}".format(csv_name))

            items_per_class = len(prediction_probabilities) // N_CLASSES # it works b/c test dataset length is divisible by N_CLASSES

            csv_name  = 'submission_' + model_name + '_by_probability.csv'
            with open(csv_name, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['fname','camera'])
                sum_prediction_probabilities_by_class = np.sum(prediction_probabilities, axis=(0,))
                for class_idx in np.argsort(sum_prediction_probabilities_by_class)[::-1]:
                    largest_idx = np.argpartition(prediction_probabilities[:,class_idx], -items_per_class)[-items_per_class:]
                    prediction_probabilities[largest_idx] = 0.
                    ids_by_class = [ids[largest_id] for largest_id in largest_idx]
                    for largest_id in ids_by_class:
                        csv_writer.writerow([largest_id.split('/')[-1], CLASSES[class_idx]])
            print("Predictions assuming prior flat probability distribution on test dataset:")
            print("kg submit {}".format(csv_name))
            csvfile.close()
