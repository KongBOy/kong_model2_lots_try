import tensorflow as tf
import numpy as np

writer = tf.summary.create_file_writer("try13_tboard")
y = np.array([3, 5, 7, 8, 9, 1, 0, 1, 1, 1, 2, 3, 5])
step = len(y)

for i in range(step):
    with writer.as_default():
        tf.summary.scalar("lr", y[i], step=i + 1)   ### 把lr模擬值 寫入 tensorboard
