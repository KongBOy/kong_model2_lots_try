import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

############################################################################################################
### batch_size > 1 的情況：bs有規律，sb完全打亂
############################################################################################################
n_data = 15
db1 = tf.data.Dataset.range(n_data)
db2 = tf.data.Dataset.range(n_data, 0, -1)

# for data in db1: print(data)  ### 0 ~ 9
# for data in db2: print(data)  ### 10 ~ 1

db_combine = tf.data.Dataset.zip((db1, db2))  ### 通常資料都是 data, label 的形式，所以這邊為了模擬這樣的形式就 先 zip 再 來做測試囉！

db_combine_sb = tf.data.Dataset.shuffle(db_combine, buffer_size=n_data ).batch(3)  ### 先shuffle 再 batch
db_combine_bs = tf.data.Dataset.shuffle(db_combine.batch(3), buffer_size=n_data)   ### 先batch 再 zip

for bs in db_combine_bs: print(bs)
### 會以 batch包起來的資料包 來 shuffle，所以可以看到資料有 012, 345, ... 的規律性
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([0, 1, 2], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([15, 14, 13], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([3, 4, 5], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([12, 11, 10], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([12, 13, 14], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([3, 2, 1], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([6, 7, 8], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([9, 8, 7], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([ 9, 10, 11], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([6, 5, 4], dtype=int64)>)
print("")

for sb in db_combine_sb: print(sb)
### 真正的 打亂 囉！
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([11, 14,  6], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([4, 1, 9], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([ 7, 10,  5], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([ 8,  5, 10], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 0, 9], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([13, 15,  6], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 8, 4], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([14,  7, 11], dtype=int64)>)
# (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([12,  3, 13], dtype=int64)>, <tf.Tensor: shape=(3,), dtype=int64, numpy=array([ 3, 12,  2], dtype=int64)>)
print("")

############################################################################################################
### batch_size == 1 的情況：都一樣是完全打亂的狀態
############################################################################################################
n_data = 10
db1 = tf.data.Dataset.range(n_data)
db2 = tf.data.Dataset.range(n_data, 0, -1)
db_combine = tf.data.Dataset.zip((db1, db2))  ### 通常資料都是 data, label 的形式，所以這邊為了模擬這樣的形式就 先 zip 再 來做測試囉！

db_combine_sb = tf.data.Dataset.shuffle(db_combine, buffer_size=n_data ).batch(1)  ### 先shuffle 再 batch
db_combine_bs = tf.data.Dataset.shuffle(db_combine.batch(1), buffer_size=n_data)   ### 先batch 再 zip
for bs in db_combine_bs: print(bs)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([3], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([7], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([1], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([9], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([0], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([10], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([5], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([5], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([6], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([4], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([6], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([9], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([1], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([2], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([8], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([8], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([2], dtype=int64)>)
print("")
for sb in db_combine_sb: print(sb)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([8], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([2], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([9], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([1], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([5], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([5], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([4], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([6], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([6], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([4], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([2], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([8], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([1], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([9], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([7], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([3], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([0], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([10], dtype=int64)>)
# (<tf.Tensor: shape=(1,), dtype=int64, numpy=array([3], dtype=int64)>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7], dtype=int64)>)
