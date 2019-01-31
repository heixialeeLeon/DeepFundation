import numpy as np
from numpy import random
import torch
from torchvision import transforms
import cv2

def Rotate(img, degree):
    rows, cols, _= img.shape
    m = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
    img = cv2.warpAffine(img, m,(cols,rows))
    return img

def Rotate_step_test(img):
    for alpha  in np.arange(-5, 5, 2):
        cv2.imshow(f"{alpha}", Rotate(img, alpha))
    cv2.waitKey(-1)

if __name__ == "__main__":
    if __name__ == "__main__":
        img = cv2.imread("../test/1.jpg")
        cv2.imshow("raw", img)
        Rotate_step_test(img)