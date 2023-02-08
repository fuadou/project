from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     MaxPooling2D, ZeroPadding2D,concatenate,Input,AveragePooling2D,Dropout)
from tensorflow.keras import backend as K
import numpy as np


CONV_BLOCK_COUNT = 0  # 用来命名计数卷积编号
INCEPTION_A_COUNT = 0
INCEPTION_B_COUNT = 0
INCEPTION_C_COUNT = 0


def conv_block(x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
    global CONV_BLOCK_COUNT
    CONV_BLOCK_COUNT += 1
    with K.name_scope('conv_block_' + str(CONV_BLOCK_COUNT)):
        x = Conv2D(filters=nb_filters,
                   kernel_size=(nb_row, nb_col),
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias)(x)
        x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
        x = Activation("relu")(x)
    return x


def stem(x_input):
    with K.name_scope('stem'):
        x = conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv_block(x, 32, 3, 3, padding='valid')
        x = conv_block(x, 64, 3, 3)

        x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

        x = concatenate([x1, x2], axis=-1)

        x1 = conv_block(x, 64, 1, 1)
        x1 = conv_block(x1, 96, 3, 3, padding='valid')

        x2 = conv_block(x, 64, 1, 1)
        x2 = conv_block(x2, 64, 7, 1)
        x2 = conv_block(x2, 64, 1, 7)
        x2 = conv_block(x2, 96, 3, 3, padding='valid')

        x = concatenate([x1, x2], axis=-1)

        x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

        merged_vector = concatenate([x1, x2], axis=-1)
    return merged_vector


def inception_A(x_input):
    """35*35 卷积块"""
    global INCEPTION_A_COUNT
    INCEPTION_A_COUNT += 1
    with K.name_scope('inception_A' + str(INCEPTION_A_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
            x_input)  # 35 * 35 * 192
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 96, 1, 1)  # 35 * 35 * 96

        conv1x1 = conv_block(x_input, 96, 1, 1)  # 35 * 35 * 96

        conv1x1_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
        conv1x1_3x3 = conv_block(conv1x1_3x3, 96, 3, 3)  # 35 * 35 * 96

        conv3x3_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
        conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
        conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96

        merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3],
                                    axis=-1)  # 35 * 35 * 384
    return merged_vector


def inception_B(x_input):
    """17*17 卷积块"""
    global INCEPTION_B_COUNT
    INCEPTION_B_COUNT += 1
    with K.name_scope('inception_B' + str(INCEPTION_B_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 128, 1, 1)

        conv1x1 = conv_block(x_input, 384, 1, 1)

        conv1x7_1x7 = conv_block(x_input, 192, 1, 1)
        conv1x7_1x7 = conv_block(conv1x7_1x7, 224, 1, 7)
        conv1x7_1x7 = conv_block(conv1x7_1x7, 256, 1, 7)

        conv2_1x7_7x1 = conv_block(x_input, 192, 1, 1)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 192, 1, 7)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 7, 1)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 1, 7)
        conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 256, 7, 1)

        merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
    return merged_vector


def inception_C(x_input):
    """8*8 卷积块"""
    global INCEPTION_C_COUNT
    INCEPTION_C_COUNT += 1
    with K.name_scope('Inception_C' + str(INCEPTION_C_COUNT)):
        averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
        averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 256, 1, 1)

        conv1x1 = conv_block(x_input, 256, 1, 1)

        # 用 1x3 和 3x1 替代 3x3
        conv3x3_1x1 = conv_block(x_input, 384, 1, 1)
        conv3x3_1 = conv_block(conv3x3_1x1, 256, 1, 3)
        conv3x3_2 = conv_block(conv3x3_1x1, 256, 3, 1)

        conv2_3x3_1x1 = conv_block(x_input, 384, 1, 1)
        conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 448, 1, 3)
        conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 512, 3, 1)
        conv2_3x3_1x1_1 = conv_block(conv2_3x3_1x1, 256, 3, 1)
        conv2_3x3_1x1_2 = conv_block(conv2_3x3_1x1, 256, 1, 3)

        merged_vector = concatenate(
            [averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
    return merged_vector


def reduction_A(x_input, k=192, l=224, m=256, n=384):
    with K.name_scope('Reduction_A'):
        """Architecture of a 35 * 35 to 17 * 17 Reduction_A block."""
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

        conv3x3 = conv_block(x_input, n, 3, 3, strides=(2, 2), padding='valid')

        conv2_3x3 = conv_block(x_input, k, 1, 1)
        conv2_3x3 = conv_block(conv2_3x3, l, 3, 3)
        conv2_3x3 = conv_block(conv2_3x3, m, 3, 3, strides=(2, 2), padding='valid')

        merged_vector = concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
    return merged_vector


def reduction_B(x_input):
    """Architecture of a 17 * 17 to 8 * 8 Reduction_B block."""
    with K.name_scope('Reduction_B'):
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

        conv3x3 = conv_block(x_input, 192, 1, 1)
        conv3x3 = conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')

        conv1x7_7x1_3x3 = conv_block(x_input, 256, 1, 1)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 256, 1, 7)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 7, 1)
        conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        merged_vector = concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
    return merged_vector


def GoogLeNet4(input):

    # Stem
    x = stem(input)  # 35 x 35 x 384
    feat1=x
    # 4 x Inception_A
    for i in range(4):
        x = inception_A(x)  # 35 x 35 x 384
    feat2=x
    # Reduction_A
    x = reduction_A(x, k=192, l=224, m=256, n=384)  # 17 x 17 x 1024
    feat3=x
    # 7 x Inception_B
    for i in range(7):
        x = inception_B(x)  # 17 x 17 x1024

    feat4=x
    # Reduction_B
    x = reduction_B(x)  # 8 x 8 x 1536

    feat5=x
    return feat1, feat2, feat3, feat4 ,feat5


