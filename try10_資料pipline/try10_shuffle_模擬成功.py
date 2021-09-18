### https://zhuanlan.zhihu.com/p/42417456

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

n_data = 100
db1 = tf.data.Dataset.range(n_data)
db1 = db1.batch(1)

db2 = tf.data.Dataset.range(0, n_data, -1)
db2 = db1.batch(1)

db_combine = tf.data.Dataset.zip( (db1, db2) )
amount = np.zeros(n_data)
### 方法一：shuffle 的 buffer size 設定成 跟 n_data 一樣 喔！但會爆記憶體
# db_combine = db_combine.shuffle(n_data)

# for data in db_combine:
#     amount[data[0][0]] += 1   ### 第一個[0]是解zip，第二個[0]是解batch
#     plt.scatter(np.arange(100), amount)
#     plt.show()


### 方法二：發現 好像只要 走訪 整個db，不要用db.take()， shuffle 的 buffer_size 好像除了1以外隨便設都可以，
###        要注意但數字越大，random的範圍就越大喔！所以還是大一點比較好覺得！
db_combine = db_combine.shuffle(10)
for data in db_combine:
    amount[data[0][0]] += 1
    plt.scatter(np.arange(100), amount)
    plt.show()