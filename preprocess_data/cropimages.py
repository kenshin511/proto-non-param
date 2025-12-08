import os
import pandas as pd
from PIL import Image
from shutil import copyfile

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# set paths
rootpath = '/nas/datasets/CUB_200_2011/CUB_200_2011/'
imgspath = rootpath + '/images/'
croppath = rootpath + 'cub200_cropped/images/'
trainpath = rootpath + 'cub200_cropped/train_cropped/'
testpath = rootpath + 'cub200_cropped/test_cropped/'

# read img names, bounding_boxes
names = pd.read_table(rootpath + 'images.txt', delimiter=' ', names=['id', 'name'])
names = names.to_numpy()
boxs = pd.read_table(rootpath + 'bounding_boxes.txt', delimiter=' ',names=['id', 'x', 'y', 'width', 'height'])
boxs = boxs.to_numpy()

folders = pd.read_table(rootpath + 'classes.txt', delimiter=' ', names=['id', 'folder'])
folders = folders.to_numpy()

for i in range(200):
    makedir(croppath + folders[i][1])

# crop imgs
for i in range(11788):
    im = Image.open(imgspath + names[i][1])
    im = im.crop((boxs[i][1], boxs[i][2], boxs[i][1] + boxs[i][3], boxs[i][2] + boxs[i][4]))
    im.save(croppath + names[i][1], quality=95)
    print('{} imgs cropped and saved.'.format(i + 1))
print('All Done.')

# mkdir for cropped imgs
for i in range(200):
    makedir(trainpath + folders[i][1]) #200
    makedir(testpath + folders[i][1])

# split imgs
labels = pd.read_table(rootpath + 'train_test_split.txt', delimiter=' ', names=['id', 'label'])
labels = labels.to_numpy()
for i in range(11788):
    if(labels[i][1] == 1):
        copyfile(croppath + names[i][1], trainpath + names[i][1])
    else:
        copyfile(croppath + names[i][1], testpath + names[i][1])
    print('{} imgs splited.'.format(i + 1))
print('All Done.')
