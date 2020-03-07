import tensorflow as tf 
from tensorflow.keras import layers
class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
print("linear_layer",linear_layer)
y = linear_layer(x)
print(y)