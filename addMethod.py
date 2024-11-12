import cv2
import numpy as np

# Image Evaluation using PSNR

def solPSNR(gt, h):
    MSE = 0
    tempArr = (gt - h)**2
    MSE = np.mean(tempArr)
    log = float(np.log(255**2/MSE)/np.log(10))
    return float(10*log)

I_h = cv2.imread('method2.png', 0)
I_h = np.array(I_h, dtype=float)
height, width = I_h.shape

I_gt = cv2.imread('HR.png', 0)
I_gt = np.array(I_gt, dtype=float)
I_l = cv2.resize(I_gt, (height//4, width//4))

img = cv2.imread('method1.png', 0)
img = np.array(img, dtype=float)

# Code for Guided Filtering
ep=0.001
size = (8, 8)
mean_I = cv2.blur(I_h, size)
mean_p = cv2.blur(img, size)
corr_I = cv2.blur(I_h*I_h, size)
corr_p = cv2.blur(I_h*img, size)
var_I = corr_I - mean_I*mean_I
cov_p = corr_p - mean_I*mean_p
a = cov_p/(var_I + ep)
b = mean_p - a*mean_I
mean_a = cv2.blur(a, size)
mean_b = cv2.blur(b, size)
q = a*I_h + mean_b
q= np.clip(q, 0, 255)

q= cv2.bilateralFilter(np.float32(q), 10, 25, 25)

print("Result PSNR:", solPSNR(I_gt, q))
cv2.imwrite("addMethod.png", q)