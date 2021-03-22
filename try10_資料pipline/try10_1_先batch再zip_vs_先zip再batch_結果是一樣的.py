import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

n_data = 10
db1 = tf.data.Dataset.range(n_data)
db2 = tf.data.Dataset.range(n_data, 0, -1)

for data in db1: print(data)  ### 0 ~ 9
for data in db2: print(data)  ### 10 ~ 1


db_combine_zb = tf.data.Dataset.zip( (db1, db2) ).batch(3)             ### 先zip 再 batch
db_combine_bz = tf.data.Dataset.zip( (db1.batch(3), db2.batch(3)) )    ### 先batch 再 zip

for zb in db_combine_zb: print(zb)
print("###################################")
for bz in db_combine_bz: print(bz)
