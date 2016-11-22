import PIL
from PIL import Image
import numpy as np

import glob
from multiprocessing import Process
import os
import re

from scipy.misc import imread
import matplotlib.pyplot as plt
from pathlib import Path


SIZE = 244
TRI_DIR = Path('annotations/trimaps')
IMG_DIR = Path('images')
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
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + .05).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def resize_image(img, size):
    return img


def prep_images(paths, out_dir):
    """

    :param paths: paths to images
    :param out_dir: directory to write outputs to
    :return: nothing
    """
    for count, path in enumerate(paths):
        if count % 100 == 0:
            print(path)
        ext = os.path.splitext(str(path))[-1].lower()
        if ext == ".jpg":
            img = Image.open(path)
            img_nrm = norm_image(img)
            img_res = resize_image(img_nrm, SIZE)
            basename = os.path.basename(str(path))
            path_out = os.path.join(str(out_dir), str(basename))
            img_res.save(path_out)
        else:
            print("Weird extension: {}".format(path))



def main():
    """Main program for running from command line"""

    # Get the paths to all the image files
    tri_img = [x for x in TRI_DIR.iterdir() if TRI_DIR.is_dir()]
    train_img = [x for x in IMG_DIR.iterdir() if IMG_DIR.is_dir()]



    # Make the output directories
    base_out = Path(BASE_DIR, 'data{}'.format(SIZE))
    train_dir_out = Path(base_out, 'train')
    test_dir_out = Path(base_out, 'test')
    os.makedirs(str(train_dir_out), exist_ok=True)
    os.makedirs(str(test_dir_out), exist_ok=True)

    procs = dict()
    procs[1] = Process(target=prep_images, args=(train_img,
                                                 train_dir_out, ))
    procs[1].start()
    procs[1].join()






if __name__ == '__main__':
    main()