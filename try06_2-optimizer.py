import tensorflow as tf 


grad = tf.constant([1.,2.,3.], dtype=tf.float64)
w = tf.Variable([0.,0.,0.],dtype=tf.float64)
w2 = tf.Variable([0.,0.,0.],dtype=tf.float64)
w3 = tf.Variable([0.,0.,0.],dtype=tf.float64)
print(grad)
print(w)
print("")

optimizer = tf.keras.optimizers.Adam()

### optimizer要小心，給兩個model要分開導傳遞，用就要create兩個optimizer才對喔！
optimizer.apply_gradients( [ [grad, w] ] )
print(w)
optimizer.apply_gradients( [ [grad, w] ] )
print(w)
optimizer.apply_gradients( [ [grad, w] ] )
print(w)
optimizer.apply_gradients( [ [grad, w] ] )
print(w)
print("")

### 可以看到下面的數值，儘管grad一樣，但是呼叫的時機點不一樣時，值就會跟上面不一樣！
optimizer.apply_gradients( [ [grad, w2] ] )
print(w2)
optimizer.apply_gradients( [ [grad, w2] ] )
print(w2)
optimizer.apply_gradients( [ [grad, w2] ] )
print(w2)
optimizer.apply_gradients( [ [grad, w2] ] )
print(w2)
print("")

### 新增一個optimizer，就會一樣了！
optimizer3 = tf.keras.optimizers.Adam()
optimizer3.apply_gradients( [ [grad, w3] ] )
print(w3)
optimizer3.apply_gradients( [ [grad, w3] ] )
print(w3)
optimizer3.apply_gradients( [ [grad, w3] ] )
print(w3)
optimizer3.apply_gradients( [ [grad, w3] ] )
print(w3)
print("")