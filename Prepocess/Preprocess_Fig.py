import skimage.io as io
from skimage import  color,transform
import numpy as np
import matplotlib
from skimage import data_dir


def resize_img(f,**kwargs):
    img = io.imread(f)  # 依次读取rgb图片
    #gray = color.rgb2gray(img)  # 将rgb图片转换成灰度图
    dst = transform.resize(img, (224, 224))  # 将灰度图片大小转换为256*256
    return dst

str = '/Users/arrow/Documents/feret-images/*.tif'
coll = io.ImageCollection(str, load_func=resize_img)
#print(len(coll))        # 显示collection里面的图片数量
#print(coll[100].shape)  # the image size
#print(type(coll[0]))    # class 'numpy.ndarray'
#io.imshow(coll[0])     # 显示某张图片
for j in range(200):    # 200 samples
    io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/left_15/' + np.str(4 + j*9) + '.jpg',
                  coll[4 + j*9]) #循环保存图片

    '''
    while j == 2:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/left_45/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 3:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/left_30/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 4:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/left_15/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 5:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/right_15/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 6:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/right_30/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 7:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/right_45/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
    while j == 8:
        for i in range(200):
            io.imsave('/Users/arrow/Documents/DataSets/Diversity_Angel/right_60/' + np.str(j + j * 9) + '.jpg',
                  coll[j + j * 9])
        str1 = '/Users/arrow/Documents/DataSets/Diversity_Angel/right_60/*.jpg'
        coll1 = io.ImageCollection(str1)
        print('The right_60 dataset contain pictures num:' + len(coll1))
        print('The picture type:' + type(coll1[0]))
    '''