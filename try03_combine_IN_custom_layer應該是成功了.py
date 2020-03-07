### 客製化layer：https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization


def instance_norm(in_x, name="instance_norm"):
    depth = in_x.get_shape()[3]
    scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
    offset = tf.Variable(tf.zeros(shape=[depth]))

    mean, variance = tf.nn.moments(in_x, axes=[0,1,2], keepdims=True)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (in_x-mean)*inv
    return scale*normalized + offset

class IN(tf.keras.layers.Layer):
    def __init__(self):
        super(IN, self).__init__()
        # self.num_outputs = num_outputs

    ### build的用途是 可以在 當Layer被使用到的時候，再來建立要訓練的權重
    ### 舉例：... = Dense(10, activation="relu")(x)
    ###     第一個參數就是 num_outputs
    ###     後面的(x) 就是 輸入的東西，通常是 layer相關的物件，對應的是下面 def call(...) 的 參數input
    ###     可以根據這個 x再來 建立要訓練的權重
    def build(self, input_shape):
        # print("input_shape",input_shape)
        depth = input_shape[-1]
        # self.scale  = self.add_variable("scale", shape = [depth], initializer=tf.random_normal_initializer( mean=1.0, stddev=0.02 ))
        self.scale  = self.add_variable("scale", shape = [depth], initializer=tf.constant_initializer(1.0)) ### debug用，真正使用時用上面那行 
        self.offset = self.add_variable("ofset", shape = [depth], initializer=tf.constant_initializer(0.0) )

        
    def call(self, input):
        # mean, variance = tf.nn.moments(input, axes=[0,1,2], keepdims=True)  ### 如果是用BN應該就是 axes用 012 囉！
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)  
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)


# from tensorflow.keras.models import Model
in_img  = tf.keras.layers.Input(shape=[3,3,1])
in_layer = IN()

# model   = tf.keras.models.Model(in_img, feature)


img  = np.arange(3*3*1,dtype = np.float32).reshape(1,3,3,1)
img2 = np.array( [ [5,7,3],[5,4,7],[2,8,9]  ] , dtype=np.float32 ).reshape(1,3,3,1)
img3 = np.array( [img.reshape(3,3,1), img2.reshape(3,3,1)] )
print((img-img.mean())/img.std())
print((img2-img2.mean())/img2.std())
print((img3-img3.mean())/img3.std())
print( "in_layer(img)" , in_layer(img) )
print( "in_layer(img2)", in_layer(img2) )
print( "in_layer(img3)", in_layer(img3) )
print("finish")