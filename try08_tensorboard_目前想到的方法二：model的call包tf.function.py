### 客製化layer：https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


tf.keras.backend.set_floatx('float16') ### 這步非常非常重要！用了才可以加速！

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
        self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.random_normal_initializer( mean=1.0, stddev=0.02 ), dtype=tf.float16)
        # self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.constant_initializer(1.0), dtype=tf.float16) ### debug用，真正使用時用上面那行 
        self.offset = self.add_weight("ofset", shape = [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float16)

        
    def call(self, input):
        # mean, variance = tf.nn.moments(input, axes=[0,1,2], keepdims=True)  ### 如果是用BN應該就是 axes用 012 囉！
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)  
        epsilon = tf.constant(1e-5,dtype=tf.float16)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, ks=3, s=1):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.conv_1 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c1   = InstanceNorm_kong()
        self.conv_2 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c2   = InstanceNorm_kong()
    
    def call(self, input_tensor):
        p = int( (self.ks-1)/2 )
        x = tf.pad( input_tensor, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        x = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_2(x)
        x = self.in_c2(x)
        return x + input_tensor


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


class Generator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Generator, self).__init__(**kwargs)
        self.conv1   = Conv2D(64  ,   kernel_size=7, strides=1, padding="valid")
        self.conv2   = Conv2D(64*2,   kernel_size=3, strides=2, padding="same")
        self.conv3   = Conv2D(64*4,   kernel_size=3, strides=2, padding="same")
        self.in_c1   = InstanceNorm_kong()
        self.in_c2   = InstanceNorm_kong()
        self.in_c3   = InstanceNorm_kong()

        self.resb1   = ResBlock(c_num=64*4)
        self.resb2   = ResBlock(c_num=64*4)
        self.resb3   = ResBlock(c_num=64*4)
        self.resb4   = ResBlock(c_num=64*4)
        self.resb5   = ResBlock(c_num=64*4)
        self.resb6   = ResBlock(c_num=64*4)
        self.resb7   = ResBlock(c_num=64*4)
        self.resb8   = ResBlock(c_num=64*4)
        self.resb9   = ResBlock(c_num=64*4)

        self.convT1  = Conv2DTranspose(64*2, kernel_size=3, strides=2, padding="same")
        self.convT2  = Conv2DTranspose(64  , kernel_size=3, strides=2, padding="same")
        self.in_cT1  = InstanceNorm_kong()
        self.in_cT2  = InstanceNorm_kong()
        self.convRGB = Conv2D(3  ,   kernel_size=7, strides=1, padding="valid")

    def call(self, input_tensor):
        x = tf.pad(input_tensor, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")

        ### c1
        x = self.conv1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        ### c2
        x = self.conv2(x)
        x = self.in_c2(x)
        x = tf.nn.relu(x)
        ### c1
        x = self.conv3(x)
        x = self.in_c3(x)
        x = tf.nn.relu(x)

        x = self.resb1(x)
        x = self.resb2(x)
        x = self.resb3(x)
        x = self.resb4(x)
        x = self.resb5(x)
        x = self.resb6(x)
        x = self.resb7(x)
        x = self.resb8(x)
        x = self.resb9(x)

        x = self.convT1(x)
        x = self.in_cT1(x)
        x = tf.nn.relu(x)

        x = self.convT2(x)
        x = self.in_cT2(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")
        x_RGB = self.convRGB(x)
        return tf.nn.tanh(x_RGB)

class CycleGAN(tf.keras.models.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.discriminator_a = Discriminator(name="D_A")
        self.discriminator_b = Discriminator(name="D_B")
        self.generator_a2b   = Generator(name="G_A2B")
        self.generator_b2a   = Generator(name="G_B2A")

    ### 也有在這裡用過 @tf.function，但結果變得很奇怪！
    @tf.function
    def call(self, imgA, imgB):
        fake_b       = self.generator_a2b  (imgA)
        fake_b_score = self.discriminator_b(fake_b)
        fake_b_cyc_a = self.generator_b2a  (fake_b)
        real_b_score = self.discriminator_b(imgB)

        fake_a       = self.generator_b2a(imgB)
        fake_a_score = self.discriminator_a(fake_a)
        fake_a_cyc_b = self.generator_a2b  (fake_a)
        real_a_score = self.discriminator_a(imgA)

        return fake_b_score, real_b_score, \
               fake_a_score, real_a_score, \
               fake_b_cyc_a, fake_a_cyc_b

@tf.function
def mse_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float16)):
    loss = tf.reduce_mean( tf.math.square( tensor1 - tensor2 ) )
    return loss * lamb

@tf.function
def train_step(imgA, imgB, step, writer):
    with tf.GradientTape(persistent=True) as tape:
        fake_b_score, real_b_score, \
        fake_a_score, real_a_score, \
        fake_b_cyc_a, fake_a_cyc_b = cyclegan(imgA,imgB)
        ### 在@tf.function內 無法使用tf.summary.trace... 的所有function喔~~(已嘗試過)

        loss_rec_a = mse_kong(imgA, fake_b_cyc_a, lamb=tf.constant(10.,tf.float16))
        loss_rec_b = mse_kong(imgB, fake_a_cyc_b, lamb=tf.constant(10.,tf.float16))
        loss_g2d_b = mse_kong(fake_b_score, tf.ones_like(fake_b_score,dtype=tf.float16), lamb=tf.constant(1.,tf.float16))
        loss_g2d_a = mse_kong(fake_a_score, tf.ones_like(fake_b_score,dtype=tf.float16), lamb=tf.constant(1.,tf.float16))
        g_total_loss = loss_rec_a + loss_rec_b + loss_g2d_b + loss_g2d_a

        loss_da_real = mse_kong( real_a_score, tf.ones_like(real_a_score ,dtype=tf.float16),  lamb=tf.constant(1.,tf.float16) )
        loss_da_fake = mse_kong( fake_a_score, tf.zeros_like(fake_a_score,dtype=tf.float16), lamb=tf.constant(1.,tf.float16) )
        loss_db_real = mse_kong( real_b_score, tf.ones_like(real_b_score ,dtype=tf.float16),  lamb=tf.constant(1.,tf.float16) )
        loss_db_fake = mse_kong( fake_b_score, tf.zeros_like(fake_b_score,dtype=tf.float16), lamb=tf.constant(1.,tf.float16) )
        d_total_loss = (loss_da_real+loss_da_fake)/2 + (loss_db_real+loss_db_fake)/2

    grad_G = tape.gradient(g_total_loss, cyclegan.generator_b2a.  trainable_weights + cyclegan.generator_a2b.  trainable_weights)
    grad_D = tape.gradient(d_total_loss, cyclegan.discriminator_a.trainable_weights + cyclegan.discriminator_b.trainable_weights)

    optimizer_G.apply_gradients( zip(grad_G, cyclegan.generator_b2a.  trainable_weights + cyclegan.generator_a2b.  trainable_weights)  )
    optimizer_D.apply_gradients( zip(grad_D, cyclegan.discriminator_a.trainable_weights + cyclegan.discriminator_b.trainable_weights)  )
    
    with writer.as_default():
        tf.summary.scalar("gen_total_loss", tf.constant(1.0,dtype=tf.float32), step=step)
        

import time

imgs = np.ones(shape=(1,256,256,3),dtype=np.float16)
imgA = np.ones(shape=(1,256,256,3),dtype=np.float16)
imgB = np.ones(shape=(1,256,256,3),dtype=np.float16)

### @tf.function寫在 model的 call 勉強可以，只是tensorboard出來 要一個個右鍵加進去graph
writer = tf.summary.create_file_writer("logs")
tf.summary.trace_on(graph=True)
cyclegan = CycleGAN()
cyclegan(imgA,imgB)
with writer.as_default():
    tf.summary.trace_export(name="model_graph",step=0)

optimizer_D = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
optimizer_G = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

for i in range(10):
    start_time = time.time()

    train_step(imgA, imgB, tf.constant(i,dtype=tf.int64), writer)
    print("train_cost_time:",time.time()-start_time)

cyclegan.save_weights('cyclegan'    , save_format='tf') 

print("finish")