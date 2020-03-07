import tensorflow as tf 
from tensorflow.keras.layers import Conv2D

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
        self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.random_normal_initializer( mean=1.0, stddev=0.02 ), dtype=tf.float32)
        # self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.constant_initializer(1.0), dtype=tf.float32) ### debug用，真正使用時用上面那行 
        self.offset = self.add_weight("ofset", shape = [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        
    def call(self, input):
        # mean, variance = tf.nn.moments(input, axes=[0,1,2], keepdims=True)  ### 如果是用BN應該就是 axes用 012 囉！
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)  
        epsilon = tf.constant(1e-5,dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)


class Discriminator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_1 = Conv2D(64  ,   kernel_size=4, strides=2, padding="same")
        self.conv_2 = Conv2D(64*2,   kernel_size=4, strides=2, padding="same")
        self.conv_3 = Conv2D(64*4,   kernel_size=4, strides=2, padding="same")
        self.conv_4 = Conv2D(64*8,   kernel_size=4, strides=2, padding="same")
        self.conv_map = Conv2D(1   ,   kernel_size=4, strides=1, padding="same")

        self.in_c2   = InstanceNorm_kong()
        self.in_c3   = InstanceNorm_kong()
        self.in_c4   = InstanceNorm_kong()
    
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

def fun(in_obj):
    print(in_obj)

optimizer = tf.keras.optimizers.SGD()
print(optimizer)
fun(optimizer)

discriminator = Discriminator()
print(discriminator)
fun(discriminator)