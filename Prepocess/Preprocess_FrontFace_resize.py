from skimage import  io,transform
import numpy as np

def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    #gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    dst = transform.resize(img, (32, 32))  # 将灰度图片大小转换为
    return dst

str = 'E:\Datasets\Diversity_Angel\\front_face\\0.jpg'
coll = io.ImageCollection(str,load_func=resize_img)
print(coll[0].shape)
io.imsave('E:\Datasets\Diversity_Angel\\front_face_resize\\32.jpg',coll[0])

