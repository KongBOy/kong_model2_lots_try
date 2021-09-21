### shuffle 參考網頁
### https://zhuanlan.zhihu.com/p/42417456
### https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/


### 要進去 try10_資料pipline 資料夾 裡面跑 debugger 資料夾的相對位置 才會對應到喔！
import sys
sys.path.append("../kong_util")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from build_dataset_combine import Check_dir_exist_and_build_new_dir
from gif_utiil import Build_gif_from_dir

import time

def build_toy_db(n_data=100):
    db1 = tf.data.Dataset.range(n_data)
    # db1 = tf.data.Dataset.list_files("X:/0 data_dir/datasets/type8_blender_os_book/blender_os_hw512_have_bg/train/dis_imgs/*.png")
    db1 = db1.batch(1)

    db2 = tf.data.Dataset.range(n_data)
    # db2 = tf.data.Dataset.list_files("X:/0 data_dir/datasets/type8_blender_os_book/blender_os_hw512_have_bg/train/dis_imgs/*.png")
    db2 = db1.batch(1)

    db_combine = tf.data.Dataset.zip( (db1, db2) )
    return db_combine


if __name__ == "__main__":
    ###################################################################
    ### 設定參數
    n_data = 900
    shuffle_buffer_size = 200
    epochs = 3

    show_on_screen = True
    save_as_gif    = False
    ###################################################################
    db_combine = build_toy_db(n_data)
    start_time = time.time()
    #################################
    ### 方法一：shuffle 的 buffer size 設定成 跟 n_data 一樣 喔！但會爆記憶體
    # db_combine = db_combine.shuffle(n_data)

    # for data in db_combine:
    #     amount[data[0][0]] += 1   ### 第一個[0]是解zip，第二個[0]是解batch
    #     plt.scatter(np.arange(100), amount)
    #     plt.show()
    #################################
    ### 方法二：發現 好像只要 走訪 整個db，不要用db.take()， shuffle 的 buffer_size 好像除了1以外隨便設都可以，
    ###        要注意但數字越大，random的範圍就越大喔！所以還是大一點比較好覺得！
    db_combine = db_combine.shuffle(shuffle_buffer_size)

    ###################################################################
    ### 視覺化的初始化
    amount = np.zeros(n_data)  ### 用來視覺化 shuffle 取資料的 容器
    data_log = []              ### 用來視覺化 shuffle 取資料的 容器

    if(save_as_gif):
        visual_img_dir = "try10_shuffle_visual"
        Check_dir_exist_and_build_new_dir(visual_img_dir)

    if(show_on_screen):
        plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
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
            print(f"after shuffle, at step_{step} data1 == data2:{data1 == data2}")
            ax[0].set_title(f"shuffle_buffer_size:{shuffle_buffer_size}, epochs={epochs}")
            ax[0].set_ylabel("count")
            ax[0].set_xlabel("data")
            ax[0].set_ylim(0, epochs)
            ax[0].scatter(np.arange(n_data), amount, s=5)

            ax[1].set_title(f"data={data1}, step={step}")
            ax[1].set_ylabel("data")
            ax[1].set_xlabel("step")
            ax[1].plot(range(len(data_log)), data_log)

            fig.tight_layout()
            if(save_as_gif):
                plt.savefig(visual_img_dir + "/" + "%06i" % step)
            if(show_on_screen):
                plt.show()
                plt.pause(0.001)
            ax[0].cla()
            ax[1].cla()

            step += 1
    print("cost time:", time.time() - start_time)
    ###################################################################
    ### 把結果存成gif
    if(save_as_gif):
        Build_gif_from_dir(ord_dir = visual_img_dir, dst_dir = ".", gif_name = "try10_shuffle_visual")
