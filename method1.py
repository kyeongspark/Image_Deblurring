import cv2
import numpy as np

# Image Evaluation using PSNR

def solPSNR(gt, h):
    MSE = 0
    tempArr = (gt - h)**2
    MSE = np.mean(tempArr)
    log = float(np.log(255**2/MSE)/np.log(10))
    return float(10*log)


# Method 1

I_h = cv2.imread('upsampled.png', 0)
I_h = np.array(I_h, dtype=float)
height, width = I_h.shape

I_gt = cv2.imread('HR.png', 0)
I_gt = np.array(I_gt, dtype=float)
I_l = cv2.resize(I_gt, (height//4, width//4))

Max_iteration = 500
alpha = 0.00707

print("Inital PSNR:", solPSNR(I_gt, I_h))

for _ in range(0, Max_iteration):
    I_dt = cv2.resize(I_h, (height//4, width//4))
    grad = cv2.resize(I_dt - I_l, (height, width))
    I_h = I_h - alpha*grad
    I_h = np.clip(I_h, 0, 255)

print("Result PSNR:", solPSNR(I_gt, I_h))
cv2.imwrite("method1.png", I_h)

