### 客製化layer：https://www.tensorflow.org/tutorials/customization/custom_layers
import tensorflow as tf
import numpy as np

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    ### build的用途是 可以在 當Layer被使用到的時候，再來建立要訓練的權重
    ### 舉例：... = Dense(10, activation="relu")(x)
    ###     第一個參數就是 num_outputs
    ###     後面的(x) 就是 輸入的東西，通常是 layer相關的物件，對應的是下面 def call(...) 的 參數input
    ###     可以根據這個 x再來 建立要訓練的權重
    def build(self, input_shape):
        print("input_shape",input_shape)
        self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs] )

    def call(self, input):
        return tf.matmul(input, self.kernel)

in_x = tf.keras.layers.Input(shape=[3])

layer = MyDenseLayer(10)(in_x)


print(layer)                           ### 看自己建的layer：Tensor("my_dense_layer/MatMul:0", shape=(None, 10), dtype=float32)
print(tf.keras.layers.Dense(10)(in_x)) ### 看tf內定的layer：Tensor("dense/BiasAdd:0", shape=(None, 10), dtype=float32)
print(in_x)                            ### 看tf內定的layer：Tensor("input_1:0", shape=(None, 3), dtype=float32)
print(tf.Variable(tf.constant(1.)))    ### <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
print(tf.constant(2., name = "testaaa")) ### 想看看名字有沒有用，好像沒用：tf.Tensor(2.0, shape=(), dtype=float32)