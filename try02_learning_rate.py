from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

x = y = np.random.randn(32, 12)  # dummy data

ipt = Input((12,))
out = Dense(12)(ipt)
model = Model(ipt, out)


# starter_learning_rate = 0.1
# end_learning_rate = 0.01
# decay_steps = 10000
# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#     starter_learning_rate,
#     decay_steps,
#     end_learning_rate,
#     power=0.5)

# model.compile(SGD(learning_rate = 1e-4, decay=1e-2), loss='mse')


model.compile(SGD(learning_rate = 0.1), loss='mse')

lr_list = []
for iteration in range(10):
    model.train_on_batch(x, y )
    model.optimizer.lr= model.optimizer.lr - 0.001
    # print(model.optimizer.get_update())
    print("lr at iteration {}: {}".format(
            iteration + 1, model.optimizer._decayed_lr('float32').numpy()))
    lr_list.append(model.optimizer._decayed_lr("float32").numpy())
print("model.optimizer.lr",model.optimizer.lr)
plt.plot(range(10), lr_list)
plt.show()