
import keras
import cv2
import numpy as np
import keras.backend as K
from nets.yolox import Focus,SiLU
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss)
from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.utils_metrics import f_score
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
K.set_learning_phase(1)  # set learning phase

# 需根据自己情况修改1.训练好的模型路径和图像路径
weight_file_dir = './logs/cbam/best_epoch_weights.h5'
img_path = './img/ddf046.jpg'

model = load_model (weight_file_dir,custom_objects={'_dice_loss_with_CE': dice_loss_with_CE,'_f_score':f_score,'Focus':Focus,'SiLU':SiLU})
image = load_img(img_path, target_size=(256, 256))

x = img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = model.predict(x)
class_idx = np.argmax(pred[-1])

class_output = model.output[:,class_idx]
# class_output = model.output[:,:,class_idx]
# 需根据自己情况修改2. 把block5_conv3改成自己模型最后一层卷积层的名字
last_conv_layer = model.get_layer("add_11")

grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
##需根据自己情况修改3. 512是我最后一层卷基层的通道数，根据自己情况修改
for i in range(96):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(img_path)
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
# cv2.imshow( '11',img)
# img = img_to_array(image)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
cv2.imshow( '22',heatmap)
superimposed_img = cv2.addWeighted(img, 0.3, heatmap, 0.7,1)
# cv2.imshow('Grad-cam', superimposed_img)
cv2.imwrite('./img_out/2d.png',superimposed_img)
cv2.waitKey(0)