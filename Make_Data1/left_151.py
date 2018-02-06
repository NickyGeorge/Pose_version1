import skimage.io as io
from skimage import  color,transform
import numpy as np

def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    #gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    dst = transform.resize(img, (224, 224, 1))  # 将灰度图片大小转换为256*256
    return dst

str = 'E:\Datasets\Diversity_Angel\left_15\*.jpg'
coll = io.ImageCollection(str, load_func=resize_img)

imgs =  np.zeros((len(coll), 224, 224, 1))
#imgs_rep = np.zeros((len(coll), 224, 224, 1))
def Collect_Pic():
    for i in range(len(coll)):
        imgs[i] = coll[i]
    return imgs
#imgs_rep = Collect_Pic()
#print(len(imgs_rep))
#print(imgs_rep[0])