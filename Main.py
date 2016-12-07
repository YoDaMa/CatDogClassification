import PIL
from PIL import Image
import numpy as np

import PreProcessing

from scipy.misc import imread
import matplotlib.pyplot as plt
from pathlib import Path

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

BASE_DIR = Path('.')

if not Path('train_data.txt').is_file() and not Path('label_data.txt').is_file():
    (train_data, label_data) = PreProcessing.main()
    train_data.tofile('train_data.txt')
    label_data.tofile('label_data.txt')
else:
    train_data = np.fromfile('train_data.txt')
    label_data = np.fromfile('label_data.txt')

