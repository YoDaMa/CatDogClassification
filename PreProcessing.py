import PIL
from PIL import Image
import numpy as np
from time import clock

import glob
from multiprocessing import Process
from multiprocessing import Queue
import os, re, cv2, random

from scipy.misc import imread
import matplotlib.pyplot as plt
from pathlib import Path



SIZE = 128
CHANNELS = 3
TRI_DIR = Path('../CatDogDataSet/annotations/trimaps')
IMG_DIR = Path('../CatDogDataSet/images')
BASE_DIR = Path('.')


# get all the names of the trimaps images.



# tri = imread(p.resolve())
# tri = imread('Abyssinian_3.png')
# plt.imshow(tri)
# plt.show()




# Inspiration: https://www.kaggle.com/gauss256/ ...
# dogs-vs-cats-redux-kernels-edition/preprocess-images

def norm_image(img):
    """
     Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1),
    and applies a [1%, 99%] contrast stretch
    """
    # YCbCr allows for adjustment of luma component (Y)
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])

    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    # Now between 0 and 1.
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + .05).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def resize_image(img, size, imtype='RGB'):
    """
    Resize PIL Image
    Resizes the image to be square with sidelength size. Pads with black.
    """
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = round(size * n_x / n_y)
    else:
        n_x_new = size
        n_y_new = round(size * n_y / n_x)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.ANTIALIAS)
    # Pad the borders to create a square image
    if imtype == 'RGB':
        img_pad = Image.new(imtype, (size, size), (128, 128, 128))
    else:
        img_pad = Image.new(imtype, (size, size), 2)
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_train_images(paths, out_dir):
    """
    :param paths: paths to images
    :param out_dir: directory to write outputs to
    :return: nothing
    """
    count = len(paths)
    data = np.ndarray((count, CHANNELS, SIZE, SIZE), dtype=np.uint8)
    for i, path in enumerate(paths):
        # print("Train:", i)
        if i % 100 == 0:
            print("Processed: {} of {}".format(i, count))
        ext = os.path.splitext(str(path))[-1].lower()
        if ext == ".jpg" or ext == ".png":
            img = Image.open(path)
            img_nrm = norm_image(img)
            img_res = resize_image(img_nrm, SIZE)
            img_mat = np.asarray(img_res, dtype=np.uint8)
            img_mat = np.transpose(img_mat)
            data[i] = img_mat
            basename = os.path.basename(str(path))
            path_out = os.path.join(str(out_dir), str(basename))
            img_res.save(path_out)

        else:
            print("Weird extension: {}".format(path))
    return data


def prep_label_images(paths, out_dir):
    """

    :param paths: paths to images
    :param out_dir: directory to write outputs to
    :return: nothing
    """
    count = len(paths)
    data = np.ndarray((count, 1, SIZE, SIZE), dtype=np.uint8)
    for i, path in enumerate(paths):
        # print("Train:", i)
        if i % 100 == 0:
            print("Processed: {} of {}".format(i, count))
        ext = os.path.splitext(str(path))[-1].lower()
        if ext == ".jpg" or ext == ".png":
            img = Image.open(path)
            img_res = resize_image(img, SIZE, 'L')
            img_mat = np.asarray(img_res, dtype=np.uint8)
            img_mat = np.transpose(img_mat)
            data[i] = img_mat
            basename = os.path.basename(str(path))
            path_out = os.path.join(str(out_dir), str(basename))
            img_res.save(path_out)

        else:
            print("Weird extension: {}".format(path))
    return data


# def readlist():
#     """
#     This method should be used to read the list.txt file in the annotations
#     directory.
#     :return: nothing.
#     """
#     pathToList = Path('../CatDogDataSet/annotations/list.txt')
#     f = open(str(pathToList),'r')
#     for i in f.readlines():
#         print(i)
#

def main():
    """Main program for running from command line"""

    # Get the paths to all the image files
    # tri_img = [x for x in TRI_DIR.iterdir() if TRI_DIR.is_dir()]
    # print(type(tri_img[0]))
    # train_img = [x for x in IMG_DIR.iterdir() if IMG_DIR.is_dir()]
    # test_img = [x for x in tri_img if '._' not in str(x)]

    pathToList = Path('../CatDogDataSet/annotations/list.txt')
    f = open(str(pathToList), 'r')
    fList = f.readlines()
    catList = []
    dogList = []
    pictureList = []
    for i in fList:
        iList = i.split()
        if '#' not in iList[0]:
            pictureList.append(iList[0])
            firstLetter = iList[0][0]
            if firstLetter.islower():
                dogList.append(iList)
            else:
                catList.append(iList)

    randIndexes = np.random.choice(len(pictureList), len(pictureList) // 100)
    pictureList = [pictureList[i] for i in randIndexes]
    train_img = [Path(IMG_DIR, "{}.jpg".format(x)) for x in pictureList]
    label_img = [Path(TRI_DIR, "{}.png".format(x)) for x in pictureList]

    # Make the output directories
    base_out = Path(BASE_DIR, 'data{}'.format(SIZE))
    train_dir_out = Path(base_out, 'train')
    label_dir_out = Path(base_out, 'label')
    os.makedirs(str(train_dir_out), exist_ok=True)
    os.makedirs(str(label_dir_out), exist_ok=True)

    train_data = prep_train_images(train_img, train_dir_out)
    label_data = prep_label_images(label_img, label_dir_out)
    print("Training Data:", train_data.shape)
    print("Label Data:", label_data.shape)
    return (train_data, label_data)

if __name__ == '__main__':
    main()
