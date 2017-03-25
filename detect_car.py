
import numpy as np
import cv2
import glob
import random

from keras import backend

def load_images(pattern):
    #load the images
    images = []
    for filename in glob.iglob(pattern):
        img = cv2.imread(filename)
        images.append(img)

    return images

#load images
x_positives = load_images('./images/real_cars/*.jpg') #y = 1
x_negatives = load_images('./images/negatives/*.jpg') #y = 0

#combine positives and negatives
y_total = [0.9 for i in x_positives] + [0.1 for i in x_negatives]
x_total = x_positives + x_negatives

#shuffle
combined = list(zip(x_total, y_total))
random.shuffle(combined)
x_total, y_total = zip(*combined)

#convert into proper array format
x_total = np.asarray(x_total)
y_total = np.asarray(y_total)

#dimensions
rows, cols = x_total.shape[1], x_total.shape[2]
input_shape = (0)

if backend.image_data_format() == 'channels_first':
    x_total.reshape(x_total.shape[0], 3, rows, cols)
    input_shape = (3, rows, cols)
else:
    x_total = x_total.reshape(x_total.shape[0], rows, cols, 3)
    input_shape = (rows, cols, 3)

# convert data from 8-bit to float
# ONLY SUPPORTS 8 BIT
x_total = x_total.astype('float32')
x_total /= 255

print('x_total shape:', x_total.shape)
