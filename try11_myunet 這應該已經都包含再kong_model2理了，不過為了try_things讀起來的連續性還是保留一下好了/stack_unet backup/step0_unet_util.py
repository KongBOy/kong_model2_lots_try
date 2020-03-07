import numpy as np
import cv2 

### 方法2：用hsv，感覺可以！
def method2(x, y, color_shift=5):       ### 最大位移量不可以超過 255，要不然顏色強度會不準，不過實際用了map來顯示發現通常值都不大，所以還加個color_shift喔~
    h, w = x.shape[:2]                  ### 影像寬高
    fx, fy = x, y                       ### u是x方向怎麼移動，v是y方向怎麼移動
    ang = np.arctan2(fy, fx) + np.pi    ### 得到運動的角度
    val = np.sqrt(fx*fx+fy*fy)          ### 得到運動的位移長度
    hsv = np.zeros((h, w, 3), np.uint8) ### 初始化一個canvas
    hsv[...,0] = ang*(180/np.pi/2)      ### B channel為 角度訊息的顏色
    hsv[...,1] = 255                    ### G channel為 255飽和度
    hsv[...,2] = np.minimum(val*color_shift, 255)   ### R channel為 位移 和 255中較小值来表示亮度，因為值最大為255，val的除4拿掉就ok了！
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) ### 把得到的HSV模型轉換為BGR顯示
    if(True):
        white_back = np.ones((h, w, 3),np.uint8)*255
        white_back[...,0] -= hsv[...,2]
        white_back[...,1] -= hsv[...,2]
        white_back[...,2] -= hsv[...,2]
    #        cv2.imshow("white_back",white_back)
        bgr += white_back
    return bgr