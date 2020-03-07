import tensorflow as tf


from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from try05_custom_model import Discriminator

a =  VGG16()
# print(type(a))
# a.summary()
# b = Xception()
# print(type(a)) ### <class 'tensorflow.python.keras.engine.training.Model'>
# print(type(d)) ### <class 'try05_custom_model.Discriminator'>，但其實
# print(type(tf.keras.models.Model())) ### <class 'tensorflow.python.keras.engine.training.Model'>

########################################################################################################################
### Layer 和 Model 還是有些差別喔！像是Model 才能使用　.summary() ！
########################################################################################################################
### 參考：
### https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
### https://github.com/tensorflow/tensorflow/issues/25036#issuecomment-542087377

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = tf.keras.layers.Dense(1)

#     def call(self, inputs, **kwargs):
#         return self.dense(inputs)

#     def model(self): ### x要丟tf.keras.layers.Input才行喔！ return 回去的是 新建立的Model，跟原始__init__建出來的幾乎一樣，只有非self的東西會新建而已，不過沒有用變數接起來也沒差，但沒事也別call太多次因為非self的東西會一直建
#         ### 測試以後建議還是在debug的時候call就好，要train的時候不要呼叫喔~~因為真的很不確定 新建出來多的東西 會不會很站記憶體，且 多建出來的東西 會不會影響訓練 過程、速度
#         x = Input(shape=(1))
#         return Model(inputs=[x], outputs=self.call(x))

# MyModel().model().summary()


########################################################################################################################
from try05_custom_model import Discriminator,CycleGAN
# imgA = tf.ones(shape=(1,244,244,3),dtype=tf.float32)
img = tf.keras.layers.Input(shape=(244,244,3),dtype=tf.float32)
imgA = tf.keras.layers.Input(shape=(244,244,3),dtype=tf.float32)
imgB = tf.keras.layers.Input(shape=(244,244,3),dtype=tf.float32)


d = Discriminator()
# print()
# d.model(img).summary()
# print()
# d(imgA)
# print()
# d.model(img).summary()
# print()
# d.summary()
# print()


# @tf.function()
# def see_board(d):
#     return d(imgA)
# writer = tf.summary.create_file_writer("see_model-summary_log")
# tf.summary.trace_on(graph=True)
# b = see_board(d)
# with writer.as_default():
#     tf.summary.trace_export(name="see_model-summary-graph",step=0)
#     writer.flush()

################################################################################################

imageA = tf.ones(shape=(1,244,244,3),dtype=tf.float32)
imageB = tf.ones(shape=(1,244,244,3),dtype=tf.float32)

c = CycleGAN()
c.model(imgA,imgB).summary()
c.model(imgA,imgB).summary()
c.model(imgA,imgB).summary(line_length=120)
@tf.function()
def see_board(c):
    return c(imageA,imageB)
writer = tf.summary.create_file_writer("see_model-summary_log")
tf.summary.trace_on(graph=True)
b = see_board(c)
with writer.as_default():
    tf.summary.trace_export(name="see_model-summary-graph",step=0)



print("finish")