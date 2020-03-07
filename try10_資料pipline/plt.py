import matplotlib.pyplot as plt 
import cv2
import matplotlib.pyplot as plt  

img = cv2.imread("13.jpg")[...,::-1] ### rgb轉bgr
cv2.imshow("img",img)
cv2.waitKey(0)

### subplot的code-style
### https://matplotlib.org/tutorials/introductory/usage.html#coding-styles
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()

### 兩張子圖以上
# fig, (ax1, ax2) = plt.subplots(1, 2) ### row=1, col=2
