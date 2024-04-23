import numpy as np
import os, glob, time
from random import shuffle
from imageio import imread
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras.backend as K

K.set_image_data_format('channels_last')
CH_AXIS = -1

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Activation, Cropping2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam

import json

INPUT_DATA = 'test_input'
OUTPUT_DATA = 'test_output'

ISIZE = 1024
NC_IN = 1
NC_OUT = 1
BATCH_SIZE = 1  # increasing the size causes to run out of memory
with open("config.json", "r") as read_file:
    config_file = json.load(read_file)
MAX_LAYERS = config_file["max_layers"]  # 1 for 16, 2 for 34, 3 for 70, 4 for 142, and 5 for 286
DISPLAY_ITERS = config_file["display_iter"]
NITERS = config_file["max_iter"]
TRIAL_NAME = f'{config_file["trial_name"]}_{NITERS}_{MAX_LAYERS}'
# TRIAL_NAME = 'TEST' + str(MAX_LAYERS) # Put this in config and use it so you can dont have to overwrite your tests

MODE = INPUT_DATA + '_to_' + OUTPUT_DATA

data_path = config_file["data_path"]
IMAGE_PATH_INPUT = data_path + '/train/' + INPUT_DATA + '/*.png'
IMAGE_PATH_OUTPUT = data_path + '/train/' + OUTPUT_DATA + '/*.png'

MODEL_PATH_MAIN = './Models/' + TRIAL_NAME + '/'
os.mkdir(MODEL_PATH_MAIN) if not os.path.exists(MODEL_PATH_MAIN) else None

MODEL_PATH = MODEL_PATH_MAIN + MODE + '/'
os.mkdir(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None

# %%

CONV_INIT = RandomNormal(0, 0.02)
GAMMA_INIT = RandomNormal(1., 0.02)


def down_conv(f, *a, **k):
    return Conv2D(f, kernel_initializer=CONV_INIT, *a, **k)


def up_conv(f, *a, **k):
    return Conv2DTranspose(f, kernel_initializer=CONV_INIT, *a, **k)


def batch_norm():
    return BatchNormalization(momentum=0.9, axis=CH_AXIS, epsilon=1.01e-5, gamma_initializer=GAMMA_INIT)


def leaky_relu(alpha):
    return LeakyReLU(alpha)


def basic_d(isize, nc_in, nc_out, max_layers):
    input_a, input_b = Input(shape=(isize, isize, nc_in)), Input(shape=(isize, isize, nc_out))
    input_full = Concatenate(axis=CH_AXIS)([input_a, input_b])

    if max_layers == 0:
        n_feature = 1
        layer = down_conv(n_feature, kernel_size=1, padding='same', activation='sigmoid')(input_full)

    else:
        n_feature = 64
        layer = down_conv(n_feature, kernel_size=4, strides=2, padding="same")(input_full)
        layer = leaky_relu(0.2)(layer)

        for i in range(1, max_layers):
            n_feature *= 2
            layer = down_conv(n_feature, kernel_size=4, strides=2, padding="same")(layer)
            layer = batch_norm()(layer, training=1)
            layer = leaky_relu(0.2)(layer)

        n_feature *= 2
        layer = ZeroPadding2D(1)(layer)
        layer = down_conv(n_feature, kernel_size=4, padding="valid")(layer)
        layer = batch_norm()(layer, training=1)
        layer = leaky_relu(0.2)(layer)

        n_feature = 1
        layer = ZeroPadding2D(1)(layer)
        layer = down_conv(n_feature, kernel_size=4, padding="valid", activation='sigmoid')(layer)

    return Model(inputs=[input_a, input_b], outputs=layer)


"""
The UNET_G function
"""

MAX_N_FEATURE = 64 * 8


def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
    assert s >= 2 and s % 2 == 0
    if nf_next is None:
        nf_next = min(nf_in * 2, MAX_N_FEATURE)
    if nf_out is None:
        nf_out = nf_in
    x = down_conv(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)), padding="same")(x)
    if s > 2:
        if use_batchnorm:
            x = batch_norm()(x, training=1)
        x2 = leaky_relu(0.2)(x)
        x2 = block(x2, s // 2, nf_next)
        x = Concatenate(axis=CH_AXIS)([x, x2])
    x = Activation("relu")(x)
    x = up_conv(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm)(x)
    x = Cropping2D(1)(x)
    if use_batchnorm:
        x = batch_norm()(x, training=1)
    if s <= 8:
        x = Dropout(0.5)(x, training=1)
    return x


def unet_g(isize, nc_in, nc_out, fixed_input_size=True):
    s = isize if fixed_input_size else None
    x = input_ = Input(shape=(s, s, nc_in))
    x = block(x, isize, nc_in, False, nf_out=nc_out, nf_next=64)
    x = Activation('tanh')(x)

    return Model(inputs=input_, outputs=[x])


# %%

net_d = basic_d(128, NC_IN, NC_OUT, MAX_LAYERS) # debug, change 120 to ISIZE

net_g = unet_g(128, NC_IN, NC_OUT) # debug, change 100 to ISIZE
real_a = net_g.input
fake_b = net_g.output
real_b = net_d.inputs[1]  # ground truth

output_d_real = net_d([real_a, real_b])
output_d_fake = net_d([real_a, fake_b])


def loss_fn(output, target):
    return -K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))


loss_d_real = loss_fn(output_d_real, K.ones_like(output_d_real))
loss_d_fake = loss_fn(output_d_fake, K.zeros_like(output_d_fake))
loss_g_fake = loss_fn(output_d_fake, K.ones_like(output_d_fake))

# pca here
def load_data(file_pattern):
    return glob.glob(file_pattern)

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = imread(os.path.join(folder_path, filename))  # Load images as grayscale
        if img is not None:
            images.append(img)
    return images

from sklearn.decomposition import PCA
folder_path = "/home/bwil0017/ir37_scratch/Stereo magnetograms vs hmi magnetograms cropped/test/test_output"
def preprocess_images(images):
    flattened_images = [img.flatten() for img in images]
    return np.stack(flattened_images)

images = load_images(folder_path)
X = preprocess_images(images)
n_components = 10  # Number of principal components
pca = PCA(n_components=n_components).fit(X)


# def loss_fn_l(output, target):
#     loss = K.mean(K.abs(output - target))
#     return loss









# import numpy as np
# from keras.layers import Layer
# from keras import backend as K
#
# class PCALayer(Layer):
#     def __init__(self, n_components, **kwargs):
#         self.n_components = n_components
#         super(PCALayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.components = self.add_weight(name='components', shape=(int(input_shape[-1]), self.n_components),
#                                           initializer='uniform', trainable=True)
#         super(PCALayer, self).build(input_shape)
#
#     def call(self, inputs):
#         centered_data = inputs - K.mean(inputs, axis=0)
#         pca_result = K.dot(centered_data, self.components)
#         return pca_result
#
# def loss_fn_l(output, target):
#     # Apply PCA transformation to output and target tensors
#     pca_layer = PCALayer(n_components=5)
#     output_pca = pca_layer(output)
#     target_pca = pca_layer(target)
#
#     # Calculate L1 distance between PCA-transformed output and target
#     loss = K.mean(K.abs(output_pca - target_pca))
#     return loss





import numpy as np
import os
from keras.preprocessing import image
from keras.layers import Layer
from keras import backend as K
from sklearn.decomposition import PCA

class PCALayer(Layer):
    def __init__(self, n_components, components=None, **kwargs):
        self.n_components = n_components
        self.components = K.variable(components) if components is not None else None
        super(PCALayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.components is None:
            raise ValueError("Principal components not provided.")
        super(PCALayer, self).build(input_shape)

    def call(self, inputs):
        centered_data = inputs - K.mean(inputs, axis=0)
        centered_data = K.reshape(centered_data, (-1, (self.components.shape)[0]))
        pca_result = K.dot(centered_data, self.components)
        return pca_result

# Load all training images from the directory
# image_dir = folder_path
# images = []
# start = (1024 - 128) // 2
# for filename in os.listdir(image_dir):
#     img = image.load_img(os.path.join(image_dir, filename))
#     img_array = image.img_to_array(img)
#     images.append(img_array)
# images = np.array(images)



# Flatten and concatenate images to create a single matrix
# images_flat = images.reshape(images.shape[0], -1)
#
# # Apply PCA to obtain principal components
# pca = PCA(n_components=2)
# pca.fit(images_flat)
components = pca.components_.T


# Define PCA layer with precomputed components
pca_layer = PCALayer(n_components=10, components=components)


def loss_fn_l(output, target):
    # Apply PCA transformation to output and target tensors
    output_pca = pca_layer(output)
    target_pca = pca_layer(target)

    # Calculate L1 distance between PCA-transformed output and target
    # loss = K.mean(K.abs(output_pca - target_pca))
    loss = K.max(output_pca)
    return loss















loss_l = loss_fn_l(fake_b,real_b)

# loss_l =  K.mean(K.abs(fake_b - real_b))
# loss_l =  np.mean(np.abs(fake_b.numpy() - real_b.numpy()))
# loss_l = np.mean(np.abs(fake_b.np() - real_b.np()))
# K.constant(np.mean(np.abs(np.array(fake_b)-np.array(real_b))))

loss_d = loss_d_real + loss_d_fake
training_updates_d = Adam(lr=2e-4, beta_1=0.5).get_updates(net_d.trainable_weights, [], loss_d)
net_d_train = K.function([real_a, real_b], [loss_d / 2.0],
                         training_updates_d)  # initialises the discriminator training process

loss_g = loss_g_fake + 100000000 * loss_l
training_updates_g = Adam(lr=2e-4, beta_1=0.5).get_updates(net_g.trainable_weights, [], loss_g)
net_g_train = K.function([real_a, real_b], [loss_g_fake, loss_l], training_updates_g)

# Get the list of input and output image paths
input_image_paths = glob.glob(IMAGE_PATH_INPUT)
print("number of images in input", len(input_image_paths))
output_image_paths = glob.glob(IMAGE_PATH_OUTPUT)
print("number of ground truth images", len(output_image_paths))





def read_image(fn, nc_in, nc_out):
    img_a = imread(fn[0], as_gray=True)
    img_b = imread(fn[1], as_gray=True)
    x, y = np.random.randint(31), np.random.randint(31)

    if nc_in != 1:
        img_a = np.pad(img_a, ((15, 15), (15, 15), (0, 0)), 'constant')
        img_a = img_a[x:x + 1024, y:y + 1024, :] / 255.0 * 2 - 1
    else:
        img_a = np.pad(img_a, 15, 'constant')
        img_a = img_a[x:x + 1024, y:y + 1024] / 255.0 * 2 - 1

    if nc_out != 1:
        img_b = np.pad(img_b, ((15, 15), (15, 15), (0, 0)), 'constant')
        img_b = img_b[x:x + 1024, y:y + 1024, :] / 255.0 * 2 - 1
    else:
        img_b = np.pad(img_b, 15, 'constant')
        img_b = img_b[x:x + 1024, y:y + 1024] / 255.0 * 2 - 1

    return img_a, img_b


def mini_batch(data_ab, batch_size, nc_in, nc_out):
    # this is a generator function
    # calling next, we return what is at yield
    length = len(data_ab)
    epoch = i = 0
    tmp_size = None
    while True:
        size = tmp_size if tmp_size else batch_size
        if i + size > length:
            shuffle(data_ab)
            i = 0
            epoch += 1
        data_a = []
        data_b = []
        for J in range(i, i + size):
            img_a, img_b = read_image(data_ab[J], nc_in, nc_out)
            data_a.append(img_a)
            data_b.append(img_b)
        data_a = np.float32(data_a)
        data_b = np.float32(data_b)
        i += size
        tmp_size = yield epoch, data_a, data_b


list_input = load_data(IMAGE_PATH_INPUT)
list_output = load_data(IMAGE_PATH_OUTPUT)

assert len(list_input) == len(list_output)
LIST_TOTAL = list(zip(sorted(list_input), sorted(list_output)))
train_batch = mini_batch(LIST_TOTAL, BATCH_SIZE, NC_IN,
                         NC_OUT)

t0 = t1 = time.time()
gen_iters = 0
err_l = 0
current_epoch = 0
err_g = 0
err_l_sum = 0
err_g_sum = 0
err_d_sum = 0

err_l_list = []
err_g_list = []
err_d_list = []



while gen_iters <= NITERS:
    current_epoch, train_a, train_b = next(
        train_batch)  # this returns the epoch number, and some other things. this just gets the next element from
    # the generator

    # input images:
    train_a = train_a.reshape((BATCH_SIZE, ISIZE, ISIZE,
                               NC_IN))  # reshape TRAIN_A to a number of separate images, with dim ISIZE^2 and NC_IN
    # channels

    # output images
    train_b = train_b.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_OUT))

    # crop
    start = (1024 - 128) // 2
    train_a = train_a[:, start:start + 128, start:start + 128, :]
    train_b = train_b[:, start:start + 128, start:start + 128, :]


    ERR_D, = net_d_train([train_a, train_b])  # update the weights then find the error
    err_d_sum += ERR_D

    err_g, err_l = net_g_train([train_a, train_b])  # update the weights then find the error
    err_g_sum += err_g
    err_l_sum += err_l

    err_d_list.append(ERR_D)
    err_g_list.append(err_g)
    err_l_list.append(err_l)

    # info
    if gen_iters % DISPLAY_ITERS == 0:
        print('[%d][%d/%d] LOSS_D: %5.3f LOSS_G: %5.3f LOSS_L: %5.3f T: %dsec/%dits, Total T: %d'
              % (
                  current_epoch, gen_iters, NITERS, err_d_sum / DISPLAY_ITERS, err_g_sum / DISPLAY_ITERS,
                  err_l_sum / DISPLAY_ITERS,
                  time.time() - t1, DISPLAY_ITERS, time.time() - t0))

        err_l_sum = 0
        err_g_sum = 0
        err_d_sum = 0
        DST_MODEL = MODEL_PATH + MODE + '_ITER' + '%07d' % gen_iters + '.h5'
        net_g.save(DST_MODEL)
        t1 = time.time()

    gen_iters += 1

plt.plot(err_l_list, label="L1")
plt.plot(err_g_list, label="G")
plt.plot(err_d_list, label="D")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()
os.mkdir('./Figures/' + TRIAL_NAME) if not os.path.exists('./Figures/' + TRIAL_NAME) else None
plt.savefig('./Figures/' + TRIAL_NAME + '/' + "loss.pdf")





# idea use PCA for the loss only. Make the loss function do as follows: fit pca to the training output. pca transform the true and generated outputs, then find the loss.