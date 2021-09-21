import tensorflow as tf

width = 100
height = 200
img = tf.ones(shape=[1, height, width, 3])

x = tf.range(start=0, limit=width, dtype=tf.float32)
x = tf.reshape(x, [1, -1])
x = tf.tile(x, [height, 1])
x = tf.expand_dims(x, axis=-1)
x = x / (width - 1)
x = x * 2 - 1
# print(x)

y = tf.range(start=0, limit=height, dtype=tf.float32)
y = tf.reshape(y, [-1, 1])
y = tf.tile(y, [1, width])
y = tf.expand_dims(y, axis=-1)
y = y / (height - 1)
y = y * 2 - 1
# print(y)

yx = tf.concat([y, x], axis=-1)
yx = tf.expand_dims(yx, axis=0)
# yx = tf.cast(yx, tf.float32)
# print(yx)

img = tf.concat([img, yx], axis=-1)
print(img)
