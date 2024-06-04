import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/content/sampleSubmission.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data_set = "dogs-vs-cats"

import zipfile 
with zipfile.ZipFile("/kaggle/input/"+ data_set +"/train.zip","r") as z:
    z.extractall(".")
    # save all files to kaggle/files/images
    destination = '/kaggle/files/images'
    z.extractall(destination)

data_ = pd.DataFrame({'file': os.listdir('/kaggle/files/images/train')})

Y=[]
for i in os.listdir('/kaggle/files/images/train'):
    if 'dog' in i:
        Y.append(1)
    else:
        Y.append(0)
        
data_['class'] = Y
file = data_['file']  

Y = data_['class']

import matplotlib.image as mpimg
from skimage.feature import hog
from skimage import data, exposure
from skimage.transform import rescale, resize

data_size = 1200
ptr=0
X = []
for i in file:
    img = mpimg.imread('/kaggle/files/images/train/' + i)
    resized_img = resize(img, (128, 64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel = True)
    X.append(fd)
    ptr = ptr+1
    if(ptr >= data_size):
        break

Y = Y[:data_size]

# dividing X, y into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
c = 1

from sklearn.svm import LinearSVC
svm_LinearSVC = LinearSVC(C=c).fit(X_train, y_train) 

from sklearn.svm import SVC
svm_svc = SVC(kernel='linear', C=c).fit(X_train, y_train)

accuracy = svm_LinearSVC.score(X_test, y_test)
print('svm_LinearSVC accuracy:', str(accuracy))

#svm_svc accuracy
accuracy = svm_svc.score(X_test, y_test)
print('svm_svc accuracy:', str(accuracy))
print("c = ", c)