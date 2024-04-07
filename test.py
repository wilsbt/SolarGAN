import os
import glob
from imageio import imread, imsave
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
import time
import tensorflow as tf
import json

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_data_format('channels_last')
channel_axis = -1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

with open("config.json", "r") as read_file:
    config_file = json.load(read_file)

max_layers = config_file["max_layers"]
mode = 'test_input_to_test_output'
trial_name = f"TEST{max_layers}"

SLEEP_TIME = 10
DISPLAY_ITER = config_file["display_iter"]
MAX_ITER = config_file["max_iter"]

INPUT = 'test_input'  # input used while training
INPUT1 = 'test_input1'  # testing input with testing output (near side data)
INPUT2 = 'test_input2'  # testing input without testing output (far side data)
OUTPUT = 'test_output'  # testing output for INPUT1

ISIZE = 1024
NC_IN = 1
NC_OUT = 1
BATCH_SIZE = 1

RSUN = 392
SATURATION = 100
THRESHOLD = 10

# %%

OP1 = INPUT1 + '_to_' + OUTPUT
OP2 = INPUT2 + '_to_' + OUTPUT

data_path = config_file["data_path"]

IMAGE_PATH1 = data_path + '/test/' + INPUT1 + '/*.png'
IMAGE_PATH2 = data_path + '/test/' + INPUT2 + '/*.png'
IMAGE_PATH3 = data_path + '/test/' + OUTPUT + '/*.png'

IMAGE_LIST1 = sorted(glob.glob(IMAGE_PATH1))
IMAGE_LIST2 = sorted(glob.glob(IMAGE_PATH2))
IMAGE_LIST3 = sorted(glob.glob(IMAGE_PATH3))

RESULT_PATH_MAIN = './Results/' + trial_name + '/'
os.mkdir(RESULT_PATH_MAIN) if not os.path.exists(RESULT_PATH_MAIN) else None

RESULT_PATH1 = RESULT_PATH_MAIN + OP1 + '/'
os.mkdir(RESULT_PATH1) if not os.path.exists(RESULT_PATH1) else None

RESULT_PATH2 = RESULT_PATH_MAIN + OP2 + '/'
os.mkdir(RESULT_PATH2) if not os.path.exists(RESULT_PATH2) else None

FIGURE_PATH_MAIN = './Figures/' + trial_name + '/'
os.mkdir(FIGURE_PATH_MAIN) if not os.path.exists(FIGURE_PATH_MAIN) else None


# %%

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def scale(data, range_in, range_out):
    domain = [range_in[0], range_out[1]]

    def interp(x):
        return range_out[0] * (1.0 - x) + range_out[1] * x

    def uninterp(x):
        B = 0
        if (domain[1] - domain[0]) != 0:
            B = domain[1] - domain[0]
        else:
            B = 1.0 / domain[1]
        return (x - domain[0]) / B

    return interp(uninterp(data))


def TUMF_value(image, R_sun, saturation, threshold):
    value_positive = 0
    value_negative = 0

    image_scale = scale(image, range_in=[0., 255.], range_out=[-saturation, saturation])

    size_x, size_y = image_scale.shape[0], image_scale.shape[1]

    for i in range(size_x):
        for j in range(size_y):
            if (i - size_x / 2) ** 2. + (j - size_y / 2) ** 2. < R_sun ** 2.:
                if image_scale[i, j] > threshold:
                    value_positive += image_scale[i, j]
                elif image_scale[i, j] < -threshold:
                    value_negative += image_scale[i, j]
                else:
                    None

    factor = (695500. / R_sun) * (695500. / R_sun) * 1000 * 1000 * 100 * 100

    flux_positive = value_positive * factor
    flux_negative = value_negative * factor
    flux_total = flux_positive + abs(flux_negative)

    return flux_positive, flux_negative, flux_total


# %%

iter = DISPLAY_ITER
while iter <= MAX_ITER:
    siter = '%07d' % iter

    model_name = './Models/' + trial_name + '/' + mode + '/' + mode + '_ITER' + siter + '.h5'
    print(model_name)
    save_path1 = RESULT_PATH1 + 'ITER' + siter + '/'
    os.mkdir(save_path1) if not os.path.exists(save_path1) else None

    save_path2 = RESULT_PATH2 + 'ITER' + siter + '/'
    os.mkdir(save_path2) if not os.path.exists(save_path2) else None

    figure_path = FIGURE_PATH_MAIN + 'ITER' + siter

    EX = 0
    while EX < 1:
        if os.path.exists(model_name):
            print('Starting Iter ' + str(iter) + ' ...')
            EX = 1
        else:
            print('Waiting Iter ' + str(iter) + ' ...')
            time.sleep(SLEEP_TIME)

    model = load_model(model_name)

    real_a = model.input
    fake_b = model.output
    net_g_generate = K.function([real_a], [fake_b])


    def net_g_gen(A):
        return np.concatenate([net_g_generate([A[i:i + 1]])[0] for i in range(A.shape[0])], axis=0)


    UTMF_real = []
    UTMF_fake = []

    for i in range(4):
    # for i in range(len(IMAGE_LIST1)):
        img = np.float32(imread(IMAGE_LIST1[i], as_gray=True) / 255.0 * 2 - 1)
        real = np.float32(imread(IMAGE_LIST3[i]), as_gray=True)
        date = IMAGE_LIST1[i][-19:-4]
        img.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        fake = net_g_gen(img)
        fake = ((fake[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        fake.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        #        SAVE_NAME = SAVE_PATH1 + OP1 + '_' + DATE + '.png'
        save_name = save_path1 + OP1 + '_' + str(i) + '.png'
        imsave(save_name, fake)

        RP, RN, RT = TUMF_value(real, RSUN, SATURATION, THRESHOLD)
        FP, FN, FT = TUMF_value(fake, RSUN, SATURATION, THRESHOLD)

        UTMF_real.append(RT)
        UTMF_fake.append(FT)

    # for j in range(len(IMAGE_LIST2)):
    for j in range(4):
        img = np.float32(imread(IMAGE_LIST2[j], as_gray=True) / 255.0 * 2 - 1)
        date = IMAGE_LIST2[j][-19:-4]
        img.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        fake = net_g_gen(img)
        fake = ((fake[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        fake.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        # SAVE_NAME = SAVE_PATH2 + OP2 + '_' + DATE + '.png'
        save_name = save_path2 + OP2 + '_' + str(j) + '.png'
        imsave(save_name, fake)






    def load_images(path, num_images):
        return [np.array(imread(path[i], as_gray=True)) for i in range(num_images)]




    INPUT1_IMAGE_LIST = listdir_nohidden(data_path + '/test/' + INPUT1 + '/')
    OUTPUT_IMAGE_LIST = listdir_nohidden(data_path + '/test/' + OUTPUT + '/')
    INPUT2_IMAGE_LIST = listdir_nohidden(data_path + '/test/' + INPUT2 + '/')
    INPUT_TRAIN_IMAGE_LIST = listdir_nohidden(data_path + '/train/' + INPUT + '/')
    OUTPUT_TRAIN_IMAGE_LIST = listdir_nohidden(data_path + '/train/' + OUTPUT + '/')
    OP1_IMAGE_LIST = listdir_nohidden(save_path1)
    OP2_IMAGE_LIST = listdir_nohidden(save_path2)



    INPUT1_IMAGES = load_images(INPUT1_IMAGE_LIST,  4)
    OUTPUT_IMAGES = load_images(OUTPUT_IMAGE_LIST,  4)
    OP1_IMAGES = load_images(OP1_IMAGE_LIST, 4)
    OP2_IMAGES = load_images(OP2_IMAGE_LIST, 2)
    INPUT2_IMAGES = load_images(INPUT1_IMAGE_LIST, 2)
    INPUT_TRAIN_IMAGES = load_images(INPUT_TRAIN_IMAGE_LIST, 2)
    OUTPUT_TRAIN_IMAGES = load_images(OUTPUT_TRAIN_IMAGE_LIST, 2)

    def add_subplots_fig2(fig, images, row_offset):
        for i, img in enumerate(images):
            ax = fig.add_subplot(3, 4, i + row_offset)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    fig2 = plt.figure()
    add_subplots_fig2(fig2, INPUT1_IMAGES, 1)
    add_subplots_fig2(fig2, OUTPUT_IMAGES, 5)
    add_subplots_fig2(fig2, OP1_IMAGES, 9)
    fig2.savefig(figure_path + '_FIGURE2.png')
    plt.close(fig2)


    CC = np.corrcoef(UTMF_real, UTMF_fake)[0, 1]
    fig3 = plt.figure()
    fig3.suptitle('CC : %6.3f' % (CC))
    plt.plot(UTMF_real, UTMF_fake, 'ro')
    fig3.savefig(figure_path + '_FIGURE3.png')
    plt.close(fig3)


    def add_subplots_fig4(fig, images, row_offset):
        for i, img in enumerate(images):
            ax = fig.add_subplot(2, 4, i + row_offset)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    fig4 = plt.figure()
    add_subplots_fig4(fig4, INPUT2_IMAGES, 1)
    add_subplots_fig4(fig4, INPUT_TRAIN_IMAGES, 3)
    add_subplots_fig4(fig4, OP2_IMAGES, 5)
    add_subplots_fig4(fig4, OUTPUT_TRAIN_IMAGES, 7)
    fig4.savefig(figure_path + '_FIGURE4.png')
    plt.close(fig4)



    del model
    K.clear_session()

    iter += DISPLAY_ITER
