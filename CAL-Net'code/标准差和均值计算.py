import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

all_file = os.listdir(r'C:\Users\fuyibin\Desktop\yuantu')  # 训练集图片所在的文件夹路径



count = 0
R_sums = []
G_sums = []
B_sums = []
for file in all_file:
    (filename, extension) = os.path.splitext(file)
    if extension == '.jpg':
        img = cv2.imread(os.path.join(r'C:\Users\fuyibin\Desktop\yuantu', file))/255
        count_i = img.shape[0] * img.shape[1]  #每张图片的像素个数
        count += count_i  # 统计所有图片的像素个数之和
        im_B = img[:, :, 0]
        im_G = img[:, :, 1]
        im_R = img[:, :, 2]
        im_B_sum = np.sum(im_B)
        im_G_sum = np.sum(im_G)
        im_R_sum = np.sum(im_R)
        B_sums.append(im_B_sum)
        G_sums.append(im_G_sum)
        R_sums.append(im_R_sum)
        print("图片 {} 的BGR像素和为：{}、{}、{}".format(file, im_B_sum, im_G_sum, im_R_sum) )
mean = [0, 0, 0]
mean[0] = np.sum(B_sums)/count
mean[1] = np.sum(G_sums)/count
mean[2] = np.sum(R_sums)/count
print('数据集的BGR均值是：{}、{}、{} '.format(mean[0], mean[1], mean[2]))


BGR_mean = [0.1574206692161545, 0.1759948413125467, 0.17366374671727164]  # 这是博主在自己的数据集中计算出来的结果，改成你自己的
count = 0
B_square_sums = []
G_square_sums = []
R_square_sums = []
for file in all_file:
    (filename, extension) = os.path.splitext(file)
    if extension == '.jpg':
        img = cv2.imread(os.path.join(r'C:\Users\fuyibin\Desktop\yuantu', file))/255
        count_i = img.shape[0] * img.shape[1]  #每张图片的像素个数
        count += count_i  # 统计所有图片的像素个数之和
        im_B = img[:, :, 0]
        im_G = img[:, :, 1]
        im_R = img[:, :, 2]

        # 计算B通道各像素与均值差的平方和
        B_square_sum = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                B_square_sum += np.square(im_B[i, j] - BGR_mean[0])
        B_square_sums.append(B_square_sum)

        # 计算G通道各像素与均值差的平方和
        G_square_sum = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                G_square_sum += np.square(im_G[i, j] - BGR_mean[1])
        G_square_sums.append(G_square_sum)

        # 计算R通道各像素与均值差的平方和
        R_square_sum = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                R_square_sum += np.square(im_R[i, j] - BGR_mean[2])
        R_square_sums.append(R_square_sum)
        print("图片 {} 的BGR像素与均值差的平方和为：{}、{}、{}".format(file, B_square_sum, G_square_sum, R_square_sum) )
std = [0, 0, 0]
std[0] = np.sqrt(np.sum(B_square_sums)/count)
std[1] = np.sqrt(np.sum(G_square_sums)/count)
std[2] = np.sqrt(np.sum(R_square_sums)/count)
print('数据集的BGR标准差是：{}、{}、{} '.format(std[0], std[1], std[2]))
