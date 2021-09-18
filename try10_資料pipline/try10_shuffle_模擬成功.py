### shuffle 參考網頁
### https://zhuanlan.zhihu.com/p/42417456
### https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/

import sys
sys.path.append("../kong_util")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from build_dataset_combine import Check_dir_exist_and_build_new_dir
from gif_utiil import Build_gif_from_dir

n_data = 100
db1 = tf.data.Dataset.range(n_data)
db1 = db1.batch(1)

db2 = tf.data.Dataset.range(0, n_data, -1)
db2 = db1.batch(1)

db_combine = tf.data.Dataset.zip( (db1, db2) )

### 用來視覺化 shuffle 取資料的 容器
amount = np.zeros(n_data)
data_log = []

### 方法一：shuffle 的 buffer size 設定成 跟 n_data 一樣 喔！但會爆記憶體
# db_combine = db_combine.shuffle(n_data)

# for data in db_combine:
#     amount[data[0][0]] += 1   ### 第一個[0]是解zip，第二個[0]是解batch
#     plt.scatter(np.arange(100), amount)
#     plt.show()


### 方法二：發現 好像只要 走訪 整個db，不要用db.take()， shuffle 的 buffer_size 好像除了1以外隨便設都可以，
###        要注意但數字越大，random的範圍就越大喔！所以還是大一點比較好覺得！

###################################################################
### 設定參數
shuffle_buffer_size = 10
epochs = 5
time_series = shuffle_buffer_size * epochs

db_combine = db_combine.shuffle(shuffle_buffer_size)
###################################################################
### 視覺化的初始化
# plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

visual_img_dir = "try10_shuffle_visual"
Check_dir_exist_and_build_new_dir(visual_img_dir)
##################################################################
## 模擬訓練時怎麼取data
step = 0
for _ in range(epochs):
    for batch_data1, batch_data2 in db_combine:
        data1 = batch_data1[0]
        data2 = batch_data2[0]
        amount[data1] += 1
        data_log.append(data1)

        ### 視覺化部分 ###################################################################
        print(f"after shuffle,at step_{step} data1 == data2:{data1 == data2}")
        ax[0].set_title(f"shuffle_buffer_size:{shuffle_buffer_size}, epochs={epochs}")
        ax[0].set_ylabel("count")
        ax[0].set_xlabel("data")
        ax[0].set_ylim(0, epochs)
        ax[0].scatter(np.arange(100), amount)

        ax[1].set_title(f"data={data1}, step={step}")
        ax[1].set_ylabel("data")
        ax[1].set_xlabel("step")
        ax[1].plot(range(len(data_log)), data_log)

        # fig.tight_layout()
        # plt.show()
        # plt.pause(0.001)
        plt.savefig(visual_img_dir + "/" + "%06i" % step)
        ax[0].cla()
        ax[1].cla()

        step += 1

###################################################################
### 把結果存成gif
Build_gif_from_dir(ord_dir = visual_img_dir, dst_dir = ".", gif_name = "try10_shuffle_visual")
