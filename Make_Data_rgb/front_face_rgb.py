import skimage.io as io
from skimage import  color,transform
import numpy as np
import PIL.Image as Image


def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    #gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    dst = transform.resize(img, (32, 32, 3))  # 将灰度图片大小转换为256*256
    return dst

    # 转换图像
    '''
    image = Image.open(f)   
    print(type(image))
    r, g, b = image.split()
    r_arr = np.array(r).reshape(1024)
    g_arr = np.array(g).reshape(1024)
    b_arr = np.array(b).reshape(1024)
    image_arr = np.concatenate((r_arr, g_arr, b_arr))
    return image_arr
    '''

str = 'E:\Datasets\Diversity_Angel_rgb\\front_face\*.jpg'
coll = io.ImageCollection(str, load_func=resize_img)

#print(coll[8])

#io.imsave('E:\Datasets\Diversity_Angel_rgb\\resize1_gray.jpg', coll[0])
imgs =  np.zeros((len(coll), 32, 32, 3))
'''
imgs_s = np.zeros((len(coll), 32*32*3, 1))
imgs_s[0] = np.reshape(coll[40], (32*32*3,1))
print(imgs_s[0])
'''
#imgs[0] = coll[0]
#print(coll[0])
#print('------------------------')
#print(imgs[0])
def Collect_Pic():
    for i in range(len(coll)):
        imgs[i] = coll[i]
    return imgs