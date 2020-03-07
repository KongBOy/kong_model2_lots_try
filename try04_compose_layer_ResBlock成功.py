### 客製化layer：https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization


class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm_kong, self).__init__()
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

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, ks=3, s=1):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.conv_1 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_1   = InstanceNorm_kong()
        self.relu   = ReLU()
        self.conv_2 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_2   = InstanceNorm_kong()
    
    def call(self, input):
        p = int( (self.ks-1)/2 )
        x = tf.pad( input, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_1(x)
        x = self.in_1(x)
        x = self.relu(x)
        x = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_2(x)
        x = self.in_2(x)
        return x + input

img  = np.arange(3*3*1,dtype = np.float32).reshape(1,3,3,1)
res_layer = ResBlock(64) ### a.1.先建立物件
print( "in_layer(img)" , res_layer(img) )    ### a.2.在使用物件
print( "in_layer(img)" , ResBlock(128)(img) ) ### b. 建立物件並馬上使用
print("finish")