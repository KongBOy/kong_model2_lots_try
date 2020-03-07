### 參考 https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/graphs.ipynb#scrollTo=zVuaKBifu-qF
import tensorflow as tf

# The function to be traced.
@tf.function
def my_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))
    #return tf.nn.relu(tf.matmul(x, y))

# Set up logging.
logdir = 'logs/func/' 
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0)