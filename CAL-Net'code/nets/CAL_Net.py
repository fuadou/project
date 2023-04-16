from nets.xception import xception
from nets.CFB import backbone1
from nets.inceptionv4 import GoogLeNet4
from nets.inceptionv2 import GoogLeNet2
from nets.inceptionv1 import GoogLeNet
from nets.yolov5 import darknet_body5
from nets.yolo7 import darknet_body7
from nets.efficient import EfficientNetB0
from nets.shufflenetv2 import ShuffleNet
from nets.ghostnet import Ghostnet
from nets.vgg import VGG16
from nets.mobilenetv3 import MobileNetV3
from nets.densenet import DenseNet
from nets.mobilenetv2 import MobileNetV2
from nets.mobilenetv1 import MobileNet
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from nets.dyconv2d import CondConv2D
from nets.vgg16 import darknet_body
from nets.resnet50 import ResNet50
import tensorflow as tf
import math
from tensorflow.keras import backend as K
def dctlfam(inputs,input_shape,filter_num,reduction_ratio=0.5):
    #channel attention
    x=inputs
    avgpool = GlobalAveragePooling2D()(x)
    maxpool = GlobalMaxPool2D()(x)

    # avg_out = Reshape((-1, 1))(avgpool)
    # avg_out = Conv1D(1, kernel_size=filter_num, padding="same",  use_bias=False, )(avg_out)
    # avg_out = Activation('sigmoid')(avg_out)
    # avg_out = Reshape((1, 1, -1))(avg_out)

    # max_out = Reshape((-1, 1))(maxpool)
    # max_out = Conv1D(1, kernel_size=filter_num, padding="same", use_bias=False, )(max_out)
    # max_out = Activation('sigmoid')(max_out)
    # max_out = Reshape((1, 1, -1))(max_out)
    Dense_layer1 = Dense(filter_num // reduction_ratio, activation='relu', )
    Dense_layer2 = Dense(filter_num, activation='relu',)
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = add([avg_out, max_out])
    channel = Activation('sigmoid', )(channel)
    channel = Reshape((1, 1, filter_num),)(channel)

    # x1=Conv2D(filter_num,(1,1),activation='relu',padding='same')(x)
    x1 = CondConv2D(filter_num, (1, 1), padding='same')(x)
    x1=Activation('relu')(x1)
    x1 = Activation('sigmoid', )(x1)

    channel_out = tf.multiply(x1, channel)

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, )
    maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, )
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    # spatial1 = Conv2D(1, (3, 3), strides=1, padding='same')(spatial)
    # spatial1 = Conv2D(1, (3, 3), strides=1, padding='same')(spatial)
    spatial1 = CondConv2D(1, (3, 3), padding='same')(spatial)
    spatial_out1 = Activation('sigmoid', )(spatial1)
    # spatial2 = Conv2D(1, (3, 3), strides=1, padding='same',dilation_rate=2)(spatial)
    # spatial2 = Conv2D(1, (5, 5), strides=1, padding='same')(spatial)
    spatial2 = CondConv2D(1, (5, 5), padding='same')(spatial)
    spatial_out2 = Activation('sigmoid', )(spatial2)
    # spatial3 = Conv2D(1, (3, 3), strides=1, padding='same',dilation_rate=3)(spatial)
    # spatial3 = Conv2D(1, (7, 7), strides=1, padding='same')(spatial)
    spatial3 = CondConv2D(1, (7, 7), padding='same')(spatial)
    spatial_out3 = Activation('sigmoid',)(spatial3)
    spatial_out=add([spatial_out1,spatial_out2,spatial_out3])

    x2=CondConv2D(filter_num,(1,1),padding='same')(channel_out)
    x2 = Activation('relu')(x2)
    # x2 = Conv2D(filter_num, (1, 1), activation='relu',padding='same')(channel_out)


    x2 = Activation('sigmoid', )(x2)
    spatial_out=tf.multiply(x2,spatial_out)

    dctlfamout = tf.multiply(channel_out, spatial_out)
    return dctlfamout


def eca_block(input_feature, b=1, gamma=2, name=""):
    channel = input_feature.shape[-1]
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = GlobalAveragePooling2D()(input_feature)

    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same",  use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, -1))(x)

    output = multiply([input_feature, x])
    return output


def se_block(input_feature, ratio=16, name=""):
    channel = input_feature.shape[-1]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)

    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros',
                       )(se_feature)

    se_feature = Dense(channel,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros',
                       )(se_feature)
    se_feature = Activation('sigmoid')(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def channel_attention(input_feature, ratio=8, name=""):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             )
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             )

    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, name=""):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          )(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, name=""):
    cbam_feature = channel_attention(cbam_feature, ratio, name=name)
    cbam_feature = spatial_attention(cbam_feature, name=name)
    return cbam_feature

def CAL_Net(input_shape=(256, 256, 3), num_classes=2, backbone="vgg", phi='tiny', weight_decay=5e-4):
    inputs = Input(input_shape)
    # -------------------------------#
    #   获得五个有效特征层
    #   feat1   512,512,64
    #   feat2   256,256,128
    #   feat3   128,128,256
    #   feat4   64,64,512
    #   feat5   32,32,512
    # -------------------------------#
    if backbone == "vgg":
        depth_dict = {'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        feat1, feat2, feat3, feat4, feat5 = darknet_body(inputs, depth, width, weight_decay=weight_decay)
    elif backbone == "MobilenetV2":
        feat1, feat2, feat3, feat4, feat5 = MobileNetV2(inputs,alpha=1.0)
    elif backbone == "resnet50":
        feat1, feat2, feat3, feat4, feat5 = ResNet50(inputs)
    elif backbone == "densenet":
        feat1, feat2, feat3, feat4, feat5 = DenseNet(inputs,backbone='densenet121')
    elif backbone == "MobilenetV1":
        feat1, feat2, feat3, feat4, feat5 = MobileNet(inputs)
    elif backbone == "MobilenetV3":
        feat1, feat2, feat3, feat4, feat5 = MobileNetV3(inputs)
    elif backbone == "vggg":
        feat1, feat2, feat3, feat4, feat5 = VGG16(inputs)
    elif backbone == "ghostnet":
        feat1, feat2, feat3, feat4, feat5 = Ghostnet(inputs)
    elif backbone == "shufflenetv2":
        feat1, feat2, feat3, feat4, feat5 = ShuffleNet(inputs,144)
    elif backbone == "efficient":
        feats, filters_outs = EfficientNetB0(inputs=inputs)
        feat1 = feats[2]
        feat2 = feats[3]
        feat3 = feats[4]
        feat4 = feats[5]
        feat5 = feats[6]

    elif backbone == "yolo7":
        transition_channels = {'tiny': 16, 'x': 40}[phi]
        block_channels = 16
        n = {'tiny': 2, 'x': 6}[phi]
        inputs = Input(input_shape)
        feat1, feat2, feat3 ,feat4,feat5= darknet_body7(inputs, transition_channels, block_channels, n, phi)
    elif backbone == "yolo5":
        depth_dict = {'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'tiny': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        inputs = Input(input_shape)
        feat1, feat2, feat3 ,feat4,feat5= darknet_body5(inputs, base_channels, base_depth, weight_decay)
    elif backbone =="inceptionv1":
        feat1, feat2, feat3, feat4, feat5 = GoogLeNet(inputs)
    elif backbone =="inceptionv2":
        feat1, feat2, feat3, feat4, feat5 = GoogLeNet2(inputs)
    elif backbone =="inceptionv4":
        feat1, feat2, feat3, feat4, feat5 = GoogLeNet4(inputs)
    elif backbone =="rfb":
        feat1, feat2, feat3, feat4, feat5 = backbone1(inputs)
    elif backbone =="xception":
        feat1, feat2, feat3, feat4, feat5, feat6 = xception(inputs)
    else:
        raise ValueError('Unsupported backbone - `{}`'.format(backbone))
    channels = [24, 48, 96, 192]
    # channels = [64, 128, 256, 512]
    # channels = [24, 48, 96, 192, 384]
    # channels = [16, 32, 64, 128]
    # channels = [16, 24, 40, 112]
    # channels = [16, 24, 40, 112,160] #ghostnet
    # channels = [24,116,232,464,1024]#shufflenetv2
    # channels = [40,80,112,192,320]# efficient
    # channels = [64,128,256,512]# yolo7
    # channels=[16,32,64,128,256]#yolo5
    # 32, 32, 512 -> 64, 64, 512
######################################################################################################################
    # P6_up = UpSampling2D(size=(2, 2))(feat6)
    # # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
    # P5 = Concatenate(axis=3)([feat5, P6_up])
    # # 64, 64, 1024 -> 64, 64, 512
    # P5 = Conv2D(channels[3], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_1 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_2 = Conv2D(channels[3], (1, 3),
    #               activation='relu',
    #               padding='same',
    #               kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_3 = Conv2D(channels[3], (3, 1),
    #               activation='relu',
    #               padding='same',
    #               kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5 = add([P5_1, P5_2, P5_3])
    #
    # P5 = Conv2D(channels[3], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_5 = Conv2D(channels[3], (1, 3),
    #               activation='relu',
    #               padding='same',
    #               kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5_6 = Conv2D(channels[3], (3, 1),
    #               activation='relu',
    #               padding='same',
    #               kernel_initializer=RandomNormal(stddev=0.02))(P5)
    # P5 = add([P5_4, P5_5, P5_6])
    # cfm0 = Concatenate(axis=3)([feat5, P5])
    # ###############################################################################################################

    P5_up = UpSampling2D(size=(2, 2))(feat5)     #rfb 去掉
    # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
    P4 = Concatenate(axis=3)([feat4, P5_up])
    # 64, 64, 1024 -> 64, 64, 512
    P4 = Conv2D(channels[3], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_1 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_2 = Conv2D(channels[3], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_3 = Conv2D(channels[3], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4 = add([P4_1, P4_2, P4_3])

    P4 = Conv2D(channels[3], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_5 = Conv2D(channels[3], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4_6 = Conv2D(channels[3], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P4)
    P4 = add([P4_4, P4_5, P4_6])

    cfm1 = Concatenate(axis=3)([feat4, P4])
    # 64, 64, 512 -> 128, 128, 512
    P4_up = UpSampling2D(size=(2, 2))(cfm1)
    # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768
    P3 = Concatenate(axis=3)([feat3, P4_up])
    # 128, 128, 768 -> 128, 128, 256
    P3 = Conv2D(channels[2], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_1 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_2 = Conv2D(channels[2], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_3 = Conv2D(channels[2], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3 = add([P3_1, P3_2, P3_3])

    # P3=se_block(P3)
    # P3=eca_block(P3)
    P3 = cbam_block(P3)
    #x2_1 = dctlfam(P3, 64, 64) #yolox-l
    # x2_1 = dctlfam(P3, 64, 96)#yolox-tiny
    # x2_2 = tf.multiply(x2_1, P3)
    # P3 = add([x2_2, P3])

    P3 = Conv2D(channels[2], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_4 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_5 = Conv2D(channels[2], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3_6 = Conv2D(channels[2], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P3)
    P3 = add([P3_4, P3_5, P3_6])
    cfm2 = Concatenate(axis=3)([feat3, P3])
    # 128, 128, 256 -> 256, 256, 256
    P3_up = UpSampling2D(size=(2, 2))(cfm2)
    # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384
    P2 = Concatenate(axis=3)([feat2, P3_up])
    # 256, 256, 384 -> 256, 256, 128

    P2 = Conv2D(channels[1], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_1 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_2 = Conv2D(channels[1], (1, 3),
                         activation='relu',
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_3 = Conv2D(channels[1], (3, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2 = add([P2_1, P2_2, P2_3])


    P2 = Conv2D(channels[1], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_4 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_5 = Conv2D(channels[1], (1, 3),
                         activation='relu',
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2_6 = Conv2D(channels[1], (3, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02))(P2)
    P2 = add([P2_4, P2_5, P2_6])

    cfm3 = Concatenate(axis=3)([feat2, P2])
    # 256, 256, 128 -> 512, 512, 128
    P2_up = UpSampling2D(size=(2, 2))(cfm3)
    # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192
    P1 = Concatenate(axis=3)([feat1, P2_up])
    # 512, 512, 192 -> 512, 512, 64
    P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_2 = Conv2D(channels[0], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_3 = Conv2D(channels[0], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1 = add([P1_1, P1_2, P1_3])

    # P1=se_block(P1)
    # P1=eca_block(P1)
    P1 = cbam_block(P1)
    # x4_1 = dctlfam(P1, 256, 24)
    # x4_1 = dctlfam(P1, 256, 16)
    # x4_2 = tf.multiply(x4_1, P1)
    # P1 = add([x4_2, P1])

    P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_5 = Conv2D(channels[0], (1, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1_6 = Conv2D(channels[0], (3, 1),
                  activation='relu',
                  padding='same',
                  kernel_initializer=RandomNormal(stddev=0.02))(P1)
    P1 = add([P1_4, P1_5, P1_6])

    cfm4 = Concatenate(axis=3)([feat1, P1])
    if backbone == "vgg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "resnet50":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone =="MobilenetV2":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    elif backbone == "densenet":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    elif backbone == "MobilenetV3":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    elif backbone == "MobilenetV1":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "vggg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "ghostnet":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "shufflenetv2":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "efficient":
        P1 = UpSampling2D(size=(8, 8))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "yolo7":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "yolo5":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "inceptionv1":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    elif backbone == "inceptionv2":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "inceptionv4":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "rfb":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "xception":
        P1 = UpSampling2D(size=(2, 2))(cfm4)
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_2 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_3 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_1, P1_2, P1_3])

        P1 = Conv2D(channels[0], 1, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_4 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
            P1)
        P1_5 = Conv2D(channels[0], (1, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1_6 = Conv2D(channels[0], (3, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(P1)
        P1 = add([P1_4, P1_5, P1_6])
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

    model = Model(inputs=inputs, outputs=P1)
    return model
