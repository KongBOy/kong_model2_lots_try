import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Flatten
from tensorflow.keras.models import Model
import pydot
import matplotlib.pyplot as plt
import numpy as np


encoder_in = Input(shape = (28,28,1))
x = Conv2D(3, kernel_size=3, strides=1, padding="same")(encoder_in)
encoder = Model(encoder_in, x)
# encoder.summary()


