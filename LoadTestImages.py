from pathlib import Path
import numpy as np
import os
from PIL import Image

SIZE = 32
CHANNELS = 3
TRI_DIR = Path('../CatDogDataSet/test_images')


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


def resize_image(img, size):
    """
    Resize PIL Image
    Resizes the image to be square with sidelength size. Pads with black.
    """
    img_res = img.resize((size, size), resample=Image.ANTIALIAS)
    return img_res


def prep_train_images(paths):
    """
    :rtype: object
    :param paths: paths to images
    :param out_dir: directory to write outputs to
    :return: nothing
    """
    count = len(paths)
    data = np.ndarray((count, SIZE, SIZE, CHANNELS), dtype=np.uint8)
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print("Processed: {} of {}".format(i, count))
        ext = os.path.splitext(str(path))[-1].lower()
        if ext == ".jpg":
            img = Image.open(path)
            img_nrm = norm_image(img)
            img_res = resize_image(img_nrm, SIZE)
            img_mat = np.asarray(img_res, dtype=np.uint8)
            data[i] = img_mat

        else:
            print("Weird extension: {}".format(path))
    data = data.reshape(count*(SIZE**2), CHANNELS)
    print(data.shape)
    return data

def main():
    test_path = [x for x in TRI_DIR.iterdir() if TRI_DIR.is_dir() and '.jpg' in str(x).lower()]
    # randIndexes = np.random.choice(len(test_path), len(test_path) // 10)
    # test_path = [test_path[i] for i in randIndexes]
    return prep_train_images(test_path)

main()