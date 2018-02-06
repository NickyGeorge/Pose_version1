import skimage.io as io
from skimage import  color,transform
import numpy as np

def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    dst = transform.resize(gray, (32, 32))  # 将灰度图片大小转换为256*256
    return dst

str = 'E:\Datasets\Diversity_Angel_rgb\\front_face\*.jpg'
coll = io.ImageCollection(str,load_func=resize_img)
print(coll[8])
#io.imsave('E:\Datasets\Diversity_Angel_rgb\\resize1_gray.jpg', coll[0])
imgs =  np.zeros((len(coll), 32, 32))
imgs[0] = coll[0]
#print(coll[0])
#print('------------------------')
#print(imgs[0])
def Collect_Pic():
    for i in range(len(coll)):
        imgs[i] = coll[i]
    return imgs