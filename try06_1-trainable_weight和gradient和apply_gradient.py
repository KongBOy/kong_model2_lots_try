import tensorflow as tf 
import numpy as np

### 建立假資料
vectors = np.array([[1.]]) 
imgs = np.ones(shape=(1,5,5,1))

##############################################################################################################
### 看看 單層Dense trainable_weight 長什麼樣子
# dense1 = tf.keras.layers.Dense(3) ### 建立 層物件
# dense1(vectors) ### 使用 層物件，讓他把要訓練的weights都建立出來
# print(len(dense1.trainable_weights)) ### 結果為 2，分別為：[w,b]
# print(    dense1.trainable_weights[0].shape) ### w的部分，shape 為 (1, 3)
# print(    dense1.trainable_weights[1].shape) ### b的部分，shape 為 (3,)

### 看看 單層Conv2D trainable_weight 長什麼樣子
# conv1 = tf.keras.layers.Conv2D(5, kernel_size=3, strides=1, padding="same")
# conv1(imgs) ### 使用 層物件，讓他把要訓練的weights都建立出來
# print(len(conv1.trainable_weights)) ### 結果為 2，分別為：[w,b]
# print(    conv1.trainable_weights[0].shape ) ### w的部分，shape 為 (3, 3, 1, 5)
# print(    conv1.trainable_weights[1].shape ) ### b的部分，shape 為 (5,)

##############################################################################################################
### 把層 包在model，看看 單層Dense trainable_weight 長什麼樣子
# class mymodel_dense(tf.keras.models.Model):
#     def __init__(self):
#         super(mymodel_dense,self).__init__()
#         self.dense1 = tf.keras.layers.Dense(3)
    
#     def call(self, input_tensor):
#         return self.dense1(input_tensor)
# model_try_dense = mymodel_dense()
# model_try_dense(vectors)
# print(len(model_try_dense.trainable_weights)) ### 結果為 2，分別為：[w,b]
# print(    model_try_dense.trainable_weights[0].shape) ### w的部分，shape 為 (1, 3)
# print(    model_try_dense.trainable_weights[1].shape) ### b的部分，shape 為 (3,)


### 把層 包在model，看看 單層Conv2D trainable_weight 長什麼樣子
# class mymodel_conv(tf.keras.models.Model):
#     def __init__(self):
#         super(mymodel_conv,self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(5, kernel_size=3, strides=1, padding="same")
    
#     def call(self, input_tensor):
#         return self.conv1(input_tensor)
# model_try_conv = mymodel_conv()
# model_try_conv(imgs)
# print(len(model_try_conv.trainable_weights)) ### 結果為 2，分別為：[w, b]
# print(    model_try_conv.trainable_weights[0].shape) ### w的部分，shape 為 (3, 3, 1, 5)
# print(    model_try_conv.trainable_weights[1].shape) ### b的部分，shape 為 (5,)


##############################################################################################################
### 把層 包在model，看看 兩層Conv2D trainable_weight 長什麼樣子
###     發現 tf.keras.models.Model　改成 tf.keras.layers.Layer，下面的 trainable_weights的部分 也是一樣喔～
###     所以 就剛好也試完 把兩層Conv2D 包在層 裡的狀況囉！
# class mymodel_conv2(tf.keras.models.Model): 
#     def __init__(self):
#         super(mymodel_conv2,self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(5, kernel_size=3, strides=1, padding="same")
#         self.conv2 = tf.keras.layers.Conv2D(5, kernel_size=3, strides=1, padding="same")
    
#     def call(self, input_tensor):
#         x = self.conv1(input_tensor)
#         return self.conv2(x)
# model_try_conv2 = mymodel_conv2()
# model_try_conv2(imgs)
# print(len(model_try_conv2.trainable_weights)) ### 結果為 4，分別為：[w1, b1, w2, b2]
# print(    model_try_conv2.trainable_weights[0].shape) ### w的部分，shape 為 (3, 3, 1, 5)
# print(    model_try_conv2.trainable_weights[1].shape) ### b的部分，shape 為 (5,)
# print(    model_try_conv2.trainable_weights[2].shape) ### w的部分，shape 為 (3, 3, 5, 5)
# print(    model_try_conv2.trainable_weights[3].shape) ### b的部分，shape 為 (5,)

##############################################################################################################
##############################################################################################################
### 測試 gradienttap、grad 和 optimizer
class mymodel_conv2_gradienttap(tf.keras.models.Model): 
    def __init__(self):
        super(mymodel_conv2_gradienttap,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(4, kernel_size=3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
    
    def call(self, input_tensor):
        x  = self.conv1(input_tensor)
        return self.conv2(x), self.conv3(x)

model_try_gradient_tap = mymodel_conv2_gradienttap() ### 建立模型
loss_fn = tf.keras.losses.MeanAbsoluteError()        ### 建立 loss_function
with tf.GradientTape() as tape:                      ### 開始記錄 gradient流 的graph
    output1, output2 = model_try_gradient_tap(imgs)  ### 影像 輸入模型，輸出output1, output2
    loss = loss_fn(imgs, output1)                    ### 拿 output1 和 imgs 做 MAE
grad = tape.gradient(loss, model_try_gradient_tap.trainable_weights) ### 用 上面記錄的 graph來 算梯度

# print(len(grad)))  ### 6，一層conv有[w,b]，所以三層conv有 [w1,b1,w2,b2,w3,b3] 共六組 grad～
# print(    grad[0]) ### conv1的w：tf.Tensor( ...         , shape=(3, 3, 1, 4), dtype=float64)
# print(    grad[1]) ### conv1的b：tf.Tensor([-0.01259228 -0.11778935  0.26165254 -0.49970844], shape=(4,), dtype=float64)
# print(    grad[2]) ### conv2的w：tf.Tensor( ...         , shape=(3, 3, 4, 1), dtype=float64)
# print(    grad[3]) ### conv2的b：tf.Tensor([-0.99999998], shape=(1,), dtype=float64)
# print(    grad[4]) ### conv3的w：None
# print(    grad[5]) ### conv3的b：None
# 因為用loss 來算grad， loss算的過程完全沒用到conv3，所以conv3的[w,b]的grad就為None

optimizer = tf.keras.optimizers.SGD(lr=1.)
optimizer.apply_gradients(zip(grad, model_try_gradient_tap.trainable_weights)) ### 正常用法　
# 丟進 apply_gradients的東西：
# [[ grad_c1w, c1w ],
#  [ grad_c1b, c1b ],
#  [ grad_c2w, c2w ],
#  [ grad_c2b, c2b ],
#  [ grad_c3w, c3w ],
#  [ grad_c3b, c3b ] ]
print(model_try_gradient_tap.trainable_weights[1])
print(model_try_gradient_tap.trainable_weights[3])
print(model_try_gradient_tap.trainable_weights[5])
print("")

### grad的tensor shape一樣，拿別人的grad套在不同人身上也行喔！
optimizer.apply_gradients(zip( [grad[3]], [model_try_gradient_tap.trainable_weights[5]]) ) 
print(model_try_gradient_tap.trainable_weights[5])

