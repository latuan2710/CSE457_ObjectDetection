# from skimage.io import imread
import glob
import os

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC

train_data = []
train_labels = []
pos_im_path = 'images/DATAIMAGE/positive/'
neg_im_path = 'images/DATAIMAGE/negative/'
model_path = 'materials/svm_models.dat'
# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(0)
train_data = np.float32(train_data)
train_labels = np.array(train_labels)
print('Data Prepared........')
print('Train Data:', len(train_data))
print('Train Labels (1,0)', len(train_labels))
print("""
Classification with SVM

""")

model = LinearSVC()
print('Training...... Support Vector Machine')
model.fit(train_data, train_labels)
joblib.dump(model, 'materials/svm_models.dat')
print('Model saved : {}'.format('materials/svm_models.dat'))
