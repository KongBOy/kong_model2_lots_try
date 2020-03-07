import os
from scipy.misc import imread
import scipy
import cv2
import numpy as np
import tensorflow as tf
# print(os.listdir("./datasets/horse2zebra/trainA"))

def load_train_data(img, load_size=286, fine_size=256):
    img = tf.image.resize(img, [load_size, load_size])
    print(img)
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    img = img[h1:h1+fine_size, w1:w1+fine_size,:]
    if np.random.random() > 0.5:
        img = tf.image.flip_left_right(img)
    print(img)
    return img



def load_all_data(img_dir):
    file_names = [ file_name for file_name in os.listdir(img_dir) if ".jpg" in file_name.lower() ]
    imgs = [ imread(img_dir + "/" + file_name, mode="RGB") for file_name in file_names ]
    return np.array(imgs)

dataA_imgs = load_all_data("./datasets/horse2zebra/trainA")
print(dataA_imgs.shape)
datasetA = tf.data.Dataset.from_tensor_slices(dataA_imgs)
datasetA = datasetA.shuffle(buffer_size = 1000)
# datasetA = datasetA.map(load_train_data)
datasetA = datasetA.prefetch(2)


dataB_imgs = load_all_data("./datasets/horse2zebra/trainB")
datasetB = tf.data.Dataset.from_tensor_slices(dataB_imgs)



# dataset = dataset.batch(30)
# counter = 0
# for go, element in enumerate(datasetA):
#     counter += 1
#     # print(element)
# print("counter",counter)

it = iter(datasetA)
cv2.imshow("it",next(it).numpy())

# print(dataset.as_numpy_iterator()) 

cv2.imshow("dataA_imgs",dataA_imgs[0])
cv2.waitKey(0)
print("finish")