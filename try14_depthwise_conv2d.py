### 圖很棒：https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_mli/doc/build/html/MLI_kernels/convolution_depthwise.html
### 圖很差但有程式碼：https://blog.csdn.net/mao_xiao_feng/article/details/78003476

import tensorflow as tf

img1 = tf.constant(value=[[[[1], [2], [3], [4]],
                           [[1], [2], [3], [4]],
                           [[1], [2], [3], [4]],
                           [[1], [2], [3], [4]]]], dtype=tf.float32)  ## shape=(1, 4, 4, 1), n, h, w, c，注意跟kernel的 h, w, c, kernel數 不一樣喔

img2 = tf.constant(value=[[[[1], [1], [1], [1]],
                           [[1], [1], [1], [1]],
                           [[1], [1], [1], [1]],
                           [[1], [1], [1], [1]]]], dtype=tf.float32)  ## shape=(1, 4, 4, 1)

img = tf.concat(values=[img1, img2], axis=3)  ### 以 img/feature 角度來看是 channel-wise concat

kernel0 = tf.constant(value=0, shape=[3, 3, 1, 1], dtype=tf.float32)  ### h, w, c, kernel數
kernel1 = tf.constant(value=1, shape=[3, 3, 1, 1], dtype=tf.float32)  ### h, w, c, kernel數
kernel2 = tf.constant(value=2, shape=[3, 3, 1, 1], dtype=tf.float32)  ### h, w, c, kernel數
kernel3 = tf.constant(value=3, shape=[3, 3, 1, 1], dtype=tf.float32)  ### h, w, c, kernel數
kernel_out1 = tf.concat(values=[kernel0, kernel1], axis=2)  ### 以kernel來看 是 channel-wise concat，結果的shape=(3, 3, 2, 1)
kernel_out2 = tf.concat(values=[kernel2, kernel3], axis=2)  ### 以kernel來看 是 channel-wise concat，結果的shape=(3, 3, 2, 1)
kernel = tf.concat(values=[kernel_out1, kernel_out2], axis=3)  ### 以kernel來看 是 kernel數 concat，結果的shape=(3, 3, 2, 2)


out_img = tf.nn.conv2d(input=img, filters=kernel, strides=1, padding="VALID")
print(out_img.shape)        ### (1, 2, 2, 2)
print(out_img[0, :, :, 0])
### [[9. 9.]
###  [9. 9.]]
print(out_img[0, :, :, 1])
### [[63. 81.]
###  [63. 81.]]

out_img2 = tf.nn.depthwise_conv2d(input=img, filter=kernel, strides=[1, 1, 1, 1], padding="VALID")
print(out_img2.shape)        ### (1, 2, 2, 2)
print(out_img2[0, :, :, 0])
### [[0. 0.]
###  [0. 0.]]
print(out_img2[0, :, :, 1])
### [[36. 54.]
###  [36. 54.]]
print(out_img2[0, :, :, 2])
### [[9. 9.]
###  [9. 9.]]
print(out_img2[0, :, :, 3])
### [[27. 27.]
###  [27. 27.]]
