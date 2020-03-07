import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Flatten
from tensorflow.keras.models import Model
import pydot
import matplotlib.pyplot as plt
import numpy as np
class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(InstanceNorm_kong, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float32)
        self.offset = self.add_weight("offset", shape = [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32 )

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
        epsilon = tf.constant(1e-5,dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)


class Discriminator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_1 = Conv2D(64  ,   kernel_size=4, strides=2, padding="same", name="conv1")
        self.conv_2 = Conv2D(64*2,   kernel_size=4, strides=2, padding="same", name="conv2")
        self.conv_3 = Conv2D(64*4,   kernel_size=4, strides=2, padding="same", name="conv3")
        self.conv_4 = Conv2D(64*8,   kernel_size=4, strides=2, padding="same", name="conv4")
        self.conv_map = Conv2D(1   ,   kernel_size=4, strides=1, padding="same", name="conv_map")

        self.in_c2   = InstanceNorm_kong(name="in_c2")
        self.in_c3   = InstanceNorm_kong(name="in_c2")
        self.in_c4   = InstanceNorm_kong(name="in_c2")
    
    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_2(x)
        x = self.in_c2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_3(x)
        x = self.in_c3(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_4(x)
        x = self.in_c4(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        return self.conv_map(x)



d = Discriminator()
d.compile(optimizer='adam',
              loss='mse')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
d.fit(x = tf.ones(shape = (1,256,256,3)),
      y = tf.ones(shape = (1,16,16,1)),
      callbacks=[tensorboard_callback])
# discriminator = Model(d_in,d)
# discriminator.summary()
# tf.keras.utils.plot_model(discriminator)


# logdir = "logs"
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True)

# graph_D()

# with writer.as_default():
#     tf.summary.trace_export(name="kong_model_graph", step=0)
#     writer.flush()
# tf.summary.trace_off()
