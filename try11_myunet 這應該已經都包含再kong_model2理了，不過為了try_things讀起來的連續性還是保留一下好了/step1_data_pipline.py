import tensorflow as tf

def step1_load_one_img(file_name):
    img = tf.io.read_file(file_name)
    img = tf.image.decode_jpeg(img)

    w = tf.shape(img)[1]
    w = w//2
    real_img  = img[:, :w, :]
    label_img = img[:, w:, :]

    real_img  = tf.cast(real_img, tf.float32)
    label_img = tf.cast(label_img, tf.float32)
    return label_img, real_img

@tf.function()
def step2_random_jitter(label_img, real_img):
    ### step2-1. resize 成稍微大一點
    # print("real_img.shape", tf.shape(real_img) ) 
    # print("real_img.shape", tf.shape(real_img)[0] ) 
    # print("real_img.shape", tf.shape(real_img)[0:1] ) 
    # h, w = tf.shape(real_img)[0:1] ### 不可以這樣寫，因為tf.shape(real_img)[0:1] 會傳回的是計算過程，所以只會回傳一個東西
    h = tf.shape(real_img)[0]
    w = tf.shape(real_img)[1]
    real_img  = tf.image.resize(real_img ,[h+30, w+30], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    label_img = tf.image.resize(label_img,[h+30, w+30], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

    ### step2-2. 隨機 對放大一點的結果，crop 原影像大小。作法：先concat起來，再crop(這樣只要crop一次就處理到兩張囉！)
    stacked_img = tf.stack([label_img, real_img], axis=0)
    cropped_img = tf.image.random_crop( stacked_img, size=[2, h, w, 3] )
    label_img = cropped_img[0]
    real_img  = cropped_img[1]

    ### step2-3. 有50%的機率左右反轉
    if tf.random.uniform(shape=()) > 0.5:
        label_img = tf.image.flip_left_right(label_img)
        real_img  = tf.image.flip_left_right(real_img)
        
    return label_img, real_img

def step3_normalize(label_img, real_img): ### 因為用tanh，所以把值弄到 [-1, 1]
    label_img = (label_img / 127.5) - 1
    real_img  = (real_img  / 127.5) - 1
    return label_img, real_img

def preprocess_train_img(file_name):
    label_img, real_img = step1_load_one_img(file_name)            ### 根據檔名，把圖片讀近來且把圖切開來
    label_img, real_img = step2_random_jitter(label_img, real_img) ### 算是做一點擴增，把影像放大一咪咪再random crop原影像大小、50%左右顛倒
    label_img, real_img = step3_normalize(label_img, real_img)     ### 因為用tanh，所以把值弄到 [-1, 1]
    return label_img, real_img

def preprocess_test_img(file_name):
    label_img, real_img = step1_load_one_img(file_name)            ### 根據檔名，把圖片讀近來且把圖切開來
    label_img, real_img = step3_normalize(label_img, real_img)     ### 因為用tanh，所以把值弄到 [-1, 1]
    return label_img, real_img


### 這部分就針對個別情況來寫好了，以目前資料庫很固定就是 train/test，就直接寫死在裡面囉～遇到CycleGAN的情況在自己改trainA,B/testA,B
def get_dataset(db_dir="datasets", db_name="facades", batch_size=1, data_amount=400):
    # import time
    # start_time = time.time()

    train_load_path = db_dir + "/" + db_name + "/" + "train" 
    train_db = tf.data.Dataset.list_files(train_load_path + "/" + "*.jpg")
    train_db = train_db.map(preprocess_train_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_db = train_db.shuffle( buffer_size=data_amount ) ### db內img總數
    train_db = train_db.batch( batch_size = batch_size)

    # import matplotlib.pyplot as plt
    # for item in train_db.take(5):
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].imshow((item[0]+1.)/2)
    #     ax[1].imshow((item[1]+1.)/2)
    #     plt.show()
        # print(item)

    test_load_path  = db_dir + "/" + db_name + "/" + "test" 
    test_db = tf.data.Dataset.list_files(test_load_path + "/" + "*.jpg", shuffle=False)
    test_db = test_db.map(preprocess_test_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_db = test_db.batch( batch_size = batch_size)
    # print( len(list(test_db.take(1))[0] ))
    # for batch in test_db.take(1):
    #     for item in batch:
    #         print(len(item))
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow((item[0]+1.)/2)
            # ax[1].imshow((item[1]+1.)/2)
            # plt.show()
    
    return train_db, test_db
    
    
if(__name__ == "__main__"):
    import time
    start_time = time.time()

    db_dir  = "datasets"
    db_name = "facades"
    _,_ = get_dataset(db_dir=db_dir, db_name=db_name)


    print(time.time()- start_time)
    print("finish")
    