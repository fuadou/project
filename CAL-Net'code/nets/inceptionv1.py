from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     MaxPooling2D, ZeroPadding2D,concatenate,Input,AveragePooling2D,Dropout)

def Inception(x,nb_filter):
    branch1x1 = Conv2D(nb_filter,(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2D(nb_filter,(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2D(nb_filter,(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2D(nb_filter,(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2D(nb_filter,(5,5), padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2D(nb_filter,(1,1),padding='same',strides=(1,1),name=None)(branchpool)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x

def GoogLeNet(input):

    # padding = 'same'，填充为(步长-1）/2，还可以用ZeroPadding2D((3,3))
    x = Conv2D(64,(7,7),strides=(2,2),padding='same')(input)
    feat1 = x
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(192,(3,3),strides=(1,1),padding='same')(x)
    feat2 = x
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,64)#256
    x = Inception(x,120)#480
    feat3 = x
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,128)#512
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,132)#528
    x = Inception(x,208)#832
    feat4 = x
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,208)
    x = Inception(x,256)#1024

    feat5=x
    return feat1, feat2, feat3, feat4 ,feat5