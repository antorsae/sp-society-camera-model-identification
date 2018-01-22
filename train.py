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

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=100, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-s', '--sub-batch-size', type=int, default=1, help='Number of crops from same image for each batch')
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

args = parser.parse_args()

args.preprocessed_input_path += '.pkl'

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

def get_crop(img, crop_size):
    # resize image so it's same as test images and pad it (reflect) if requested crop is bigger than image
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='reflect')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    return img[center_y - half_crop : center_y + half_crop, center_x - half_crop : center_x + half_crop]


def gen(items, batch_size, training=True, inference=False):

    images_cached = { }

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
                img = np.array(get_crop(img, 512 * 2)) # * 2 bc many need to scale by 0.5x and still get a 512px crop
                # store it in a dict for later (greatly accelerates subsequent epochs)
                images_cached[item] = img

            img =  np.array(images_cached[item])

            #print("o: ", img.shape, item)
            manipulated = 0.
            if (np.random.rand() < 0.5) and not inference:
                img = random_manipulation(img)
                #print("am: ", img.shape, item)
                manipulated = 1.

            #print("bc: ", img.shape, item)
            img = get_crop(img, CROP_SIZE)
            #print("ac: ", img.shape, item)
            img = preprocess_image(img)
            #print("ap: ", img.shape, item)

            X[batch_idx] = img
            O[batch_idx] = manipulated

            if not inference:
                y[batch_idx] = class_idx

            if batch_idx == 0 and False: #remove False if you want to see images on stdout (requires iterm2)
                show_image(X[batch_idx])

            batch_idx += 1

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
            #img = skimage.transform.resize(img, (512, 512), mode='reflect')

            img = preprocess_image(img)

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
    match = re.search(r'([A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5', args.model)
    model_name = match.group(1)
    last_epoch = int(match.group(2))
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

    x = Conv2D(3, (7, 7), strides=(1,1), use_bias=False, padding='valid', name='filtering')(image_filtered)
    x = classifier_model(x)
    if args.pooling == 'none':
        x = Flatten()(x)
    x = Dropout(args.dropout)(x)
    x = concatenate([x,manipulated])
    x = Dense(256, activation='relu')(x)
    x = Dropout(args.dropout)(x)
    x = Dense(128,  activation='relu')(x)
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

    ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)

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
            #img = np.transpose(img, axes=(1,0,2))
            img = np.expand_dims(preprocess_image(img), axis=0)

            manipulated = np.float32([[1. if idx.find('manip') != -1 else 0.]])

            prediction = model.predict_on_batch([img,manipulated])
            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = CLASSES.index(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:

                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
        if args.test_train:
            print("Accuracy: " + str(correct_predictions / i))
                