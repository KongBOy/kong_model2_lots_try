### 參考：https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/get_started.ipynb#scrollTo=TTWcJO35IJgK

### code 參考：https://github.com/tensorflow/tensorflow/issues/30270
import tensorflow as tf
from tensorflow import keras

# Load the data.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Pre-processing.
train_images = train_images / 255.0
test_images = test_images / 255.0

logdir = 'tmp/tb_test_2/'
writer = tf.summary.create_file_writer(logdir)

def instance_norm(in_x, name="instance_norm"):
    
    depth = in_x.get_shape()[3]
    scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
    offset = tf.Variable(tf.zeros(shape=[depth]))

    mean, variance = tf.nn.moments(in_x, axes=[1,2], keepdims=True)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (in_x-mean)*inv
    return scale*normalized + offset






class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128, activation='relu')
        self.bn = keras.layers.BatchNormalization()
        self.d2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn(x)
        return self.d2(x)

model = MyModel()
optimizer = keras.optimizers.Adam(0.1)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(inputs, labels, step):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        pred_loss = loss_fn(labels, predictions)
        
        with writer.as_default():
            tf.summary.scalar("loss", pred_loss, step=step)

    gradients = tape.gradient(pred_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for i in range(5):
    tf.summary.trace_on(graph=True)
    train_step(train_images, train_labels, step=i)
    

    with writer.as_default():
        tf.summary.trace_export(name="test_model", step=i)
        writer.flush()