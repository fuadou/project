

from PIL import Image

# 原图
img = Image.open('./datasets/JPEGImages/8848.png')
# 分割图
msk = Image.open('./datasets/SegmentationClass/8848.png')
# 图片格式调整一致
img = img.convert('RGBA')
msk = msk.convert('RGBA')
# 尺寸调整
img = img.resize((2484, 1632))
msk = msk.resize((2484, 1632))

image = Image.blend(img, msk, 0.6)
image.show()
