import skimage.io as io
from skimage import  color,transform
import numpy as np

def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    #gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    #dst = transform.resize(img, (224, 224, 3))  # 将灰度图片大小转换为256*256
    minValue = img.min()
    maxValue = img.max()
    dst = (img - minValue) / (maxValue - minValue)  # normalize to 0-1
    return dst

str = 'E:\Datasets\Diversity_Angel_rgb\left_15\*.jpg'
coll = io.ImageCollection(str, load_func=resize_img)
#print(coll[0])
#print(coll[0].max())
#print(coll[0]/255)
#print(coll[0].shape)
imgs =  np.zeros((len(coll), 224, 224, 3))
imgs[0] = coll[0]
#print(imgs[0])
#print('-------------------------')
#print(coll[0])
def Collect_Pic():
    for i in range(len(coll)):
        imgs[i] = coll[i]
    return imgs