import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nets.unet import Unet

model = Unet([256, 256, 3], 2)
model.summary()


def prepocess(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    # print(x.shape)
    x = tf.image.resize(x, [256,256])

    x = tf.cast(x, dtype=tf.float32)/255.
    return x

img_path='./img/ddf013.jpg'
img=prepocess(img_path)
plt.figure()
plt.imshow(img)


from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers[:245]] #前16层输出
# print(layer_outputs)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) #构建能够输出前16层的模型


input_image=tf.expand_dims(img, 0) # 扩维
activations = activation_model.predict(input_image) #12组特征层输出
print(activations[60].shape) #0对应summary表中的输入层


# 4  25  60 95  125
# cfm  138- 137 95          up  140-60 139       cfm  179- 60 178




plt.matshow(activations[233][0,:,:,0], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,1], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,2], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,3], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,4], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,5], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,6], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,7], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,8], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,9], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,10], cmap='viridis') #第1卷积层的第1特征层输出
plt.matshow(activations[233][0,:,:,11], cmap='viridis') #第1卷积层的第1特征层输出

plt.matshow(activations[233][0,:,:,95], cmap='viridis') #第1卷积层的第1特征层输出

plt.matshow(activations[60][0,:,:,1], cmap='viridis') #第1卷积层的第1特征层输出

plt.matshow(activations[95][0,:,:,1], cmap='viridis') #第1卷积层的第1特征层输出

plt.matshow(activations[125][0,:,:,1], cmap='viridis') #第1卷积层的第1特征层输出

plt.matshow(activations[4][0,:,:,1], cmap='viridis') #第1卷积层的第1特征层输出






