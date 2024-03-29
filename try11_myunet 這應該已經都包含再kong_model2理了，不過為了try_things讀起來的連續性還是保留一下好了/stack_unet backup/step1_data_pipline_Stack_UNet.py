import tensorflow as tf
import os
import numpy as np
import cv2
tf.keras.backend.set_floatx('float32')  ### 這步非常非常重要！用了才可以加速！

def step1_load_one_img(file_name):
    img = tf.io.read_file(file_name)
    img = tf.image.decode_bmp(img)
    img  = tf.cast(img, tf.float32)
    return img

def step2_resize(img):  ### h=472, w=360
    img = tf.image.resize(img , [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    return img


def step3_normalize(img):  ### 因為用tanh，所以把值弄到 [-1, 1]
    img = (img / 127.5) - 1
    return img

def preprocess_distorted_img(file_name):
    img  = step1_load_one_img(file_name)            ### 根據檔名，把圖片讀近來且把圖切開來
    img  = step2_resize(img)
    img  = step3_normalize(img)     ### 因為用tanh，所以把值弄到 [-1, 1]
    return img

def get_all_distorted(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".bmp" in file_name]
    distorted_list = []
    for file_name in file_names:
        distorted = cv2.imread(ord_dir + "/" + file_name)
        distorted = cv2.resize(distorted, (256, 256), interpolation=cv2.INTER_CUBIC)
        distorted = distorted[:, :, ::-1]
        distorted_list.append(distorted)
    distorted_list = np.array(distorted_list)
    distorted_list = (distorted_list / 127.5) - 1
    distorted_list = distorted_list.astype(np.float32)
    return distorted_list

### 可以再思考一下要不要把 get 和 resize 和 norm 拆開funtcion寫~~~ 因為直接get完直接norm，在外面就得不到原始 max min 了
def get_db_all_move_map(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".npy" in file_name]
    move_map_list = []
    for file_name in file_names:
        # print(file_name)
        move = np.load(ord_dir + "/" + file_name)
        move = cv2.resize( move, (256, 256), interpolation = cv2.INTER_NEAREST)
        move_map_list.append( move )
    move_map_list = np.array(move_map_list)
    return move_map_list

    # max_value = move_map_list.max() ###  236.52951204508076
    # min_value = move_map_list.min() ### -227.09562801056995
    # move_map_list = ((move_map_list-min_value)/(max_value-min_value))*2-1
    # return move_map_list, max_value, min_value

def use_db_to_norm(move_map_list):
    max_value = move_map_list.max()  ###  236.52951204508076
    min_value = move_map_list.min()  ### -227.09562801056995
    move_map_list = ((move_map_list - min_value) / (max_value - min_value)) * 2 - 1
    return move_map_list, max_value, min_value  ### max_value, min_value 需要回傳回去喔，因為要給 test 用

def use_number_to_norm(move_map_list, max_value, min_value):  ### 給test來用，要用和train一樣的 max_value, min_value
    move_map_list = ((move_map_list - min_value) / (max_value - min_value)) * 2 - 1
    return move_map_list

### 這部分就針對個別情況來寫好了，以目前資料庫很固定就是 train/test，就直接寫死在裡面囉～遇到CycleGAN的情況在自己改trainA,B/testA,B
def get_dataset(db_dir="datasets", db_name="stack_unet-256-100", batch_size=1):
    # import time
    # start_time = time.time()

    ### 拿到 扭曲影像 的 train dataset，從 檔名 → tensor
    distorted_train_load_path = db_dir + "/" + db_name + "/" + "train/distorted"
    distorted_train_db = get_all_distorted(distorted_train_load_path)
    distorted_train_db = tf.data.Dataset.from_tensor_slices(distorted_train_db)
    # distorted_train_db = tf.data.Dataset.list_files(distorted_train_load_path + "/" + "*.bmp", shuffle=False)
    # distorted_train_db = distorted_train_db.map(preprocess_distorted_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    distorted_train_db = distorted_train_db.batch(batch_size)

    ### 拿到 扭曲影像如何復原move 的 train dataset，從 直切先全部讀出來成npy → tensor
    rec_move_map_train_path = db_dir + "/" + db_name + "/" + "train/rec_move_map"
    rec_move_map_train_list = get_db_all_move_map(rec_move_map_train_path)
    rec_move_map_train_list, max_value_train, min_value_train = use_db_to_norm(rec_move_map_train_list)
    rec_move_map_train_db = tf.data.Dataset.from_tensor_slices(rec_move_map_train_list)
    rec_move_map_train_db = rec_move_map_train_db.batch(batch_size)

    ### 拿到 扭曲影像 的 test dataset，從 檔名 → tensor
    distorted_test_load_path = db_dir + "/" + db_name + "/" + "test/distorted"
    distorted_test_db = get_all_distorted(distorted_test_load_path)
    distorted_test_db = tf.data.Dataset.from_tensor_slices(distorted_test_db)
    # distorted_test_db = tf.data.Dataset.list_files(distorted_test_load_path + "/" + "*.bmp", shuffle=False)
    # distorted_test_db = distorted_test_db.map(preprocess_distorted_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    distorted_test_db = distorted_test_db.batch(batch_size)

    ### 拿到 扭曲影像如何復原move 的 test dataset，從 直切先全部讀出來成npy → tensor
    rec_move_map_test_path = db_dir + "/" + db_name + "/" + "test/rec_move_map"
    rec_move_map_test_list = get_db_all_move_map(rec_move_map_test_path)
    rec_move_map_test_list = use_number_to_norm(rec_move_map_test_list, max_value_train, min_value_train)
    rec_move_map_test_db = tf.data.Dataset.from_tensor_slices(rec_move_map_test_list)
    rec_move_map_test_db = rec_move_map_test_db.batch(batch_size)



    ##########################################################################################################################################
    # 在用這段的時候，記得把上面的 .batch 部分註解掉喔！
    # import matplotlib.pyplot as plt
    # from step0_unet_util import method2

    # take_num = 3
    # # for i, (img, move) in enumerate(zip(distorted_train_db.take(take_num), rec_move_map_train_db.take(take_num))): ### 想看train的部分用這行 且 註解掉下行
    # for i, (img, move) in enumerate(zip(distorted_test_db.take(take_num), rec_move_map_test_db.take(take_num))):     ### 想看test 的部分用這行 且 註解掉上行
    #     print("i",i)
    #     fig, ax = plt.subplots(1,2)
    #     img_back = (img[0]+1.)*127.5
    #     img_back = tf.cast(img_back, tf.int32)
    #     ax[0].imshow(img_back)


    #     # move_back = (move[0]+1)/2 * (max_value_train-min_value_train) + min_value_train  ### 想看train的部分用這行 且 註解掉下行
    #     move_back = (move[0]+1)/2 * (max_value_train-min_value_train) + min_value_train       ### 想看test 的部分用這行 且 註解掉上行
    #     move_bgr = method2(move_back[...,0], move_back[...,1],2)
    #     ax[1].imshow(move_bgr)
    #     plt.show()
    ##########################################################################################################################################

    return distorted_train_db, rec_move_map_train_db, distorted_test_db, rec_move_map_test_db, max_value_train, min_value_train


if(__name__ == "__main__"):
    import time
    start_time = time.time()

    db_dir  = "datasets"
    db_name = "stack_unet-256-4000"
    _ = get_dataset(db_dir=db_dir, db_name=db_name)


    print(time.time() - start_time)
    print("finish")
