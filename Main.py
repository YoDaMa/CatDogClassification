import PIL
from PIL import Image
import numpy as np

import FlatPreProcessing

from scipy.misc import imread
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


BASE_DIR = Path('.')

if not Path('train_data.txt').is_file() and not Path('label_data.txt').is_file():
    (train_data, label_data) = FlatPreProcessing.main()
    # train_data.tofile('train_data.txt')
    # label_data.tofile('label_data.txt')
    print(label_data.shape)
else:
    train_data = np.fromfile('train_data.txt')
    label_data = np.fromfile('label_data.txt')
    print(label_data.shape)

label_data = label_data.ravel()
train_data.shape

rf = RandomForestClassifier()

rf.fit(train_data, label_data)

# print("Prediction for 0: {} | {}".format(rf.predict(train_data[0:100]), label_data[0:100]))
# print("Prediction for 1: {} | {}".format(rf.predict(train_data[1]), label_data[1]))

print("Prediciting Data..")
predicitions = rf.predict(train_data)
print("Finished Random Forest!")

neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(train_data,label_data)
predictions = neigh.predict()



truepos = np.sum(np.multiply(predictions, label_data))
falsepos = np.sum(predictions) - truepos
falseneg = np.sum(label_data) - truepos
precision = truepos / (truepos+falsepos)
recall = truepos / (truepos+falseneg)
dice = 2 * precision * recall / (precision + recall)

print(dice)