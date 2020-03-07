import tensorflow as tf 
import matplotlib.pyplot as plt


def load_one_img(file_name):   ### file_name：Tensor("args_0:0", shape=(), dtype=string)
    img = tf.io.read_file(file_name) ### tf.io.read_file(filename,name=None)
    img = tf.image.decode_jpeg(img)  ### tf.image.decode_jpeg()：https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

    ### pix2pix dataset 的特性，去看一張就知道了，圖分一半是 標籤 和 GT影像
    w = tf.shape(img)[1]        ### shape 大概同numpy所以不查了
    w = w // 2
    real_image = img[:, :w, :]  ### slice 大概同numpy所以不查了
    input_image = img[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)  ### cast 就是變換dtype，補充一下：tf.float32 是可以跑很快的資料型態
    real_image  = tf.cast(real_image, tf.float32)

    return input_image, real_image


####################################################################################################################################################
### step1：定義流程：
###     以下 都是定義 計算過程喔！所以print出來都是tensor，但裡面都沒有值！要道使用 .take(...) 且用iter 才會真的拿到內容，且是根據 你定義的 計算過程 來拿到內容喔！

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, 
                                  [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) ### https://www.tensorflow.org/api_docs/python/tf/image/resize
    real_image = tf.image.resize(real_image, 
                                 [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0) ### 拚接在一起crop，就只需要crop一次就好囉！
    cropped_image = tf.image.random_crop( stacked_image, size=[2, 256, 256, 3])  ### tf.image.random_crop(value, size, seed=None, name=None)
    return cropped_image[0], cropped_image[1] ### 然後再分開傳回去

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image) ### tf.image.flip_left_right(image)
        real_image   = tf.image.flip_left_right(real_image)
    return input_image, real_image

##############################################################################
# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image  = (real_image  / 127.5) - 1

    return input_image, real_image


def transform_function(file_name):
    input_image, real_image = load_one_img(file_name)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

path = "C:/Users/TKU/.keras/datasets/facades/"

train_db = tf.data.Dataset.list_files(path+"train/*.jpg", shuffle = False)
### def list_files(file_pattern, shuffle=None, seed=None):


train_db = train_db.map( map_func= transform_function)
### def map(self, map_func, num_parallel_calls=None):
#     map_func：用來 做處理的function，目前 Dataset物件裡面的元素 會給map_func當輸入
#     num_parallel_calls：平行處理時非同步元素的數量，可設定 tf.data.experimental.AUTOTUNE 自動動態決定
#     return 回來的東西會自動包進train_db裡，並把原本的data替換掉囉！


train_db = train_db.shuffle(400)
### def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
# buffer_size：通常要設的比資料多、或等於，意思是會從db裡隨機挑這些數量的資料出來，比如有10000筆資料，buffer_size設1000，會從10000裡 隨機挑1000張出來，當使用了1000張裡的一張，該張會從buffer內拿掉，從db內剩下9000張中隨機挑一張進buffer補齊1000張
# reshuffle_each_iteration：每輪完一次dataset要不要自動shuffle，可參考下例：
'''
>>> dataset = tf.data.Dataset.range(3)
>>> dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
>>> dataset = dataset.repeat(2)  # doctest: +SKIP
[1, 0, 2, 1, 2, 0]

>>> dataset = tf.data.Dataset.range(3)
>>> dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
>>> dataset = dataset.repeat(2)  # doctest: +SKIP
[1, 0, 2, 1, 0, 2]
'''

train_db = train_db.prefetch(400)


####################################################################################################################################################
### step2：取用資料

### 無法直接取，要用iter的方式取
# print("train_db",train_db) ### <DatasetV1Adapter shapes: ((None, None, None), (None, None, None)), types: (tf.float32, tf.float32)>
# print(train_db.take(1))    ### <DatasetV1Adapter shapes: ((None, None, None), (None, None, None)), types: (tf.float32, tf.float32)>

# for item in train_db:
#     print(item)

### 用 matplotlib show 圖
for item in train_db.take(3):
    # print(item)
    fig ,ax = plt.subplots(1,2)
    ax[0].imshow(item[0]/2+0.5) ### 值要 0. ~ 1. 喔！
    ax[1].imshow(item[1]/2+0.5)
    ax[0].axis("off") ### 把x, y軸數字關掉
    ax[1].axis("off") ### 把x, y軸數字關掉
    plt.show()


### take 會從頭取，如果shuffle有打開，每次take都會洗牌
# for item in train_db.take(3):
#     print(item)
# print("")

# print( len(list(train_db)) )
# for item in train_db.take(3):
#     print(item)
# print("")
