import numpy as np
import cv2
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from numpy import random

def Hue(img, alpha):
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,0] += alpha
    img[:,:,0][img[:,:,0]>360.0] -= 360.0
    img[:,:,0][img[:,:,0]<0.0] += 360.0
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = img.astype(np.uint8)
    return img

def Huv_step_test(img):
    for alpha  in np.arange(-50, 50, 20):
        cv2.imshow(f"{alpha}", Hue(img, alpha))
    cv2.waitKey(-1)

if __name__ == "__main__":
    if __name__ == "__main__":
        img = cv2.imread("../test/1.jpg")
        cv2.imshow("raw", img)

        # img_huv = Hue(img,18)
        # cv2.imshow("huv", img_huv)
        # cv2.waitKey(-1)

        Huv_step_test(img)