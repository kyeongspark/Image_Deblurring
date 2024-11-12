import cv2
import numpy as np

# Image Evaluation using PSNR

def solPSNR(gt, h):
    MSE = 0
    tempArr = (gt - h)**2
    MSE = np.mean(tempArr)
    log = float(np.log(255**2/MSE)/np.log(10))
    return float(10*log)

# Method 2

I_h = cv2.imread('upsampled.png', 0)
I_h = np.array(I_h, dtype=float)
height, width = I_h.shape

I_gt = cv2.imread('HR.png', 0)
I_gt = np.array(I_gt, dtype=float)
I_l = cv2.resize(I_gt, (height//4, width//4))

gamma = 6

x_dir = np.absolute(cv2.Sobel(I_h, ddepth=-1, dx=1, dy=0))
y_dir = np.absolute(cv2.Sobel(I_h, ddepth=-1, dx=0, dy=1))
G_h0 = (x_dir+y_dir - np.min(x_dir+y_dir)) / (np.max(x_dir+y_dir) - np.min(x_dir+y_dir)) + 1e-10
Lap_h0 = cv2.Laplacian(I_h, ddepth=-1)
G_t = G_h0 - ((Lap_h0 - np.min(Lap_h0)) / (np.max(Lap_h0) - np.min(Lap_h0)))
G_t = np.clip(G_t, 0, 1.0)
Lap_It = gamma * Lap_h0 * G_t / G_h0
Lap_It = np.clip(Lap_It, 0.0, 255.0)
beta = 0.1

print("Inital PSNR:", solPSNR(I_gt, I_h))
Max_iteration = 1000
alpha = 0.0036

for _ in range(0, Max_iteration):
    Lap_ht = cv2.Laplacian(I_h, ddepth=-1)
    I_dt = cv2.resize(I_h, (height//4, width//4))
    grad1 = cv2.resize(I_dt - I_l, (height, width))
    grad2 = beta*((gamma*cv2.Laplacian(I_h, ddepth=-1)*G_t / G_h0) - Lap_It)
    I_h = I_h - alpha*(grad1 - grad2)
    
I_h = np.clip(I_h, 0, 255)
print("Result PSNR:", solPSNR(I_gt, I_h))

cv2.imwrite("method2.png", I_h)