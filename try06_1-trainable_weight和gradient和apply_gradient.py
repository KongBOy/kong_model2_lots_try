import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf 
import numpy as np

### 設定 random seed
tf.random.set_seed(1234)
### 建立假資料
vectors = np.array([[1.]]) 
batch_size = 1
in_h = 4
in_w = 4
imgs = np.ones(shape=(batch_size,in_h,in_w,1), dtype=np.float32) #* np.arange(2, batch_size+2, dtype=np.float32).reshape(batch_size,1,1,1)
# print(imgs)
##############################################################################################################
### 看看 單層Dense trainable_weight 長什麼樣子
# dense1 = tf.keras.layers.Dense(3) ### 建立 層物件
# dense1(vectors) ### 使用 層物件，讓他把要訓練的weights都建立出來
# print(len(dense1.trainable_weights)) ### 結果為 2，分別為：[w,b]
# print(    dense1.trainable_weights[0].shape) ### w的部分，shape 為 (1, 3)
# print(    dense1.trainable_weights[1].shape) ### b的部分，shape 為 (3,)

### 看看 單層Conv2D trainable_weight 長什麼樣子
# conv1 = tf.keras.layers.Conv2D(5, kernel_size=3, strides=1, padding="same") ### 第一個參數是 output_channel數，意思就是 要幾顆kernel
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
loss_fn = tf.keras.losses.MeanSquaredError()        ### 建立 loss_function
with tf.GradientTape() as tape:                      ### 開始記錄 gradient流 的graph
    output1, output2 = model_try_gradient_tap(imgs)  ### 影像 輸入模型，輸出output1, output2
    # loss = loss_fn(imgs, output1)                           ### 1 ###拿 output1 和 imgs 做 loss_fn(MSE)，其中沒有用到conv3，所以下面 grad_3w, b 會是 None 喔！
    # loss =                 output2 - output1                                           ### 2 ###做完會有 (bn, in_h, in_w, 1) 個值
    # loss =                (output2 - output1)**2                                       ### 3 ###做完會有 (bn, in_h, in_w, 1) 個值
    # loss = tf.reduce_sum ((output2 - output1)**2, axis=(1,2))                          ### 4 ###做完會有 (bn, 1) 個值
    # loss = tf.reduce_sum ((output2 - output1)**2, axis=(1,2))/(in_h*in_w)              ### 5 ###做完會有 (bn, 1) 個值
    # loss = (tf.reduce_sum((output2 - output1)**2, axis=(1,2))/(in_h*in_w))/batch_size  ### 6 ###做完會有 (bn, 1) 個值
    # loss = tf.reduce_mean((output2 - output1)**2, axis=(1,2))                          ### 7 ###做完會有 (bn, 1) 個值，跟上面一樣
    # loss = tf.reduce_mean((output2 - output1)**2)                                      ### 8 ###做完會有 (1) 個值，相當於把上面 bn 個值 做平均
    loss = loss_fn(output2 ,output1 )                                                    ### 9 ###做完會有 (1) 個值，跟上面一樣
print( "loss:", loss ) 

grad = tape.gradient(loss , model_try_gradient_tap.trainable_weights) ### 用 上面記錄的 graph來 算梯度
# 以下不管batch是多少，size都是長那樣子喔！感覺上就是會幫你做 sum 或 mean的感覺，這部份很重要 待確認！
# print( len(grad)  ) ### 6，一層conv有[w,b]，所以三層conv有 [w1,b1,w2,b2,w3,b3] 共六組 grad～
# print(    "grad_1w:",grad[0] ) ### conv1的w：tf.Tensor( ...         , shape=(3, 3, 1, 4), dtype=float64)
# print(    "grad_1b:",grad[1] ) ### conv1的b：tf.Tensor([-0.01259228 -0.11778935  0.26165254 -0.49970844], shape=(4,), dtype=float64)
# print(    "grad_2w:",grad[2] ) ### conv2的w：tf.Tensor( ...         , shape=(3, 3, 4, 1), dtype=float64)
# print(    "grad_2b:",grad[3] ) ### conv2的b：tf.Tensor([-0.99999998], shape=(1,), dtype=float64)
# print(    "grad_3w:",grad[4] ) ### conv3的w：None
# print(    "cnn_3b:", model_try_gradient_tap.conv3.weights[1] ) ### conv3的b：<tf.Variable 'mymodel_conv2_gradienttap/conv2d_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>
print(    "grad_3b:", grad[5] ) 
# loss = loss_fn(imgs, output1)                           ### 1 ### 的時候，conv_3b 的 grad[5] 為 None，因為loss算的過程完全沒用到conv3，所以conv3的[w,b]的grad就為None
# loss =                 output2 - output1                                           ### 2 ###的時候，當bn=1時：conv_3b 的 grad[5] =16，bn=2 的 grad[5] =16*2，之後同理 grad[5]=16*bn，代表grad會自動 把多bn自動加總
# loss =                (output2 - output1)**2                                       ### 3 ###的時候，當bn=1時：conv_3b 的 grad[5] =28.211502，和上面 不一樣，代表 對pixel 本身操作 對 grad 有影響，loss的shape為 bn, in_h, in_w
# loss = tf.reduce_sum ((output2 - output1)**2, axis=(1,2))                          ### 4 ###的時候，當bn=1時：conv_3b 的 grad[5] =28.211502，和上面  一樣 ，代表 對pixel 之間sum 對 grad 無影響，loss的shape為 bn, 1(是上面 in_h, in_w的總和) ，且 grad 和上面是一樣的，由此推斷，grad會把 (in_h, in_w) 所有的 grad值 自動加總
# loss = tf.reduce_sum ((output2 - output1)**2, axis=(1,2))/(in_h*in_w)              ### 5 ###的時候，當bn=1時：conv_3b 的 grad[5] =1.76321  ，和上面 不一樣，同第二行道理 除 batch_size 是對 pixel 本身的操作，所以對grad會有影響，bn=2 的 grad[5] =1.76321*2，之後同理 grad[5] =1.76321*bn，bn 越大 grad 越大
# loss = (tf.reduce_sum((output2 - output1)**2, axis=(1,2))/(in_h*in_w))/batch_size  ### 6 ###的時候，當bn=1時：conv_3b 的 grad[5] =1.76321，bn=2 的 grad[5] = 1.76321，之後同理都為 1.76321，不會隨著bn越大 而變大
# loss = tf.reduce_mean((output2 - output1)**2, axis=(1,2))                          ### 7 ###同第四行，相當於第二行 做 sum / (in_h*in_w)
# loss = tf.reduce_mean((output2 - output1)**2)                                      ### 8 ###同第五行，相當於第二行 做 sum / (in_h*in_w) / batch_size
loss = loss_fn(output2 ,output1 )                                                    ### 9 ###同第五行，相當於第二行 做 sum / (in_h*in_w) / batch_size
### 總結：grad 會 自動以 in_h, in_w 為單位自動加總(由第四行可知)， 再來會以 bn 為單位自動加總(由第二行可知)，以整體來看就是會對 所有值自動加總拉~~~ 但是不會幫你做平均，平均是要自己做的喔！


# optimizer = tf.keras.optimizers.SGD(lr=1.)
# optimizer.apply_gradients(zip(grad, model_try_gradient_tap.trainable_weights)) ### 正常用法　
# 丟進 apply_gradients的東西：
# [[ grad_c1w, c1w ],
#  [ grad_c1b, c1b ],
#  [ grad_c2w, c2w ],
#  [ grad_c2b, c2b ],
#  [ grad_c3w, c3w ],
#  [ grad_c3b, c3b ] ]
# print(model_try_gradient_tap.trainable_weights[1])
# print(model_try_gradient_tap.trainable_weights[3])
# print(model_try_gradient_tap.trainable_weights[5])
# print("")

### grad的tensor shape一樣，拿別人的grad套在不同人身上也行喔！
# optimizer.apply_gradients(zip( [grad[3]], [model_try_gradient_tap.trainable_weights[5]]) ) 
# print(model_try_gradient_tap.trainable_weights[5])

