import colorsys
import copy
import time

import cv2
import numpy as np
from PIL import Image

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
# from transunet.model import TransUNet

class Unet(object):
    _defaults = {

        "model_path"        : 'logs/densenet121/ep120-loss0.136-val_loss0.211.h5',

        "num_classes"       : 2,

        "backbone"          : "densenet",

        "input_shape"       : [256, 256],

        "mix_type"          : 1,
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)


    def generate(self):

        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)
        # self.model = TransUNet(256)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))


    def detect_image(self, image, count=False, name_classes=None):

        image       = cvtColor(image)

        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)


        pr = self.model.predict(image_data)[0]

        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)
        

        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image   = Image.fromarray(np.uint8(seg_img))

            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):

        image       = cvtColor(image)

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)


        pr = self.model.predict(image_data)[0]

        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
                
        t1 = time.time()
        for _ in range(test_interval):

            pr = self.model.predict(image_data)[0]

            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

            pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
    def get_miou_png(self, image):

        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)


        pr = self.model.predict(image_data)[0]

        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
