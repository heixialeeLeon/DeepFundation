import numpy as np
from numpy import random
import torch
from torchvision import transforms
import cv2

def Contrast(img, alpha):
    img = img.astype(np.float32)
    img = np.clip(alpha*img,0,255)
    img = img.astype(np.uint8)
    return img

def Brightness(img, delta):
    img = img.astype(np.float32)
    img = np.clip(img+delta,0,255)
    img = img.astype(np.uint8)
    return img

def RandomContrast(img, lower=0.5, upper=1.5):
    alpha = random.uniform(lower, upper)
    return Contrast(img, alpha)

def RandomBrightness(img, delta=32.0):
    assert delta >= 0.0
    assert delta <= 255.0
    delta = random.uniform(-delta,delta)
    return Brightness(img, delta)

if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")
    img_contrast = Contrast(img, 1.5)
    img_brightness = Brightness(img, -50)
    cv2.imshow("raw",img)
    cv2.imshow("contrast",img_contrast)
    cv2.imshow("brightness", img_brightness)
    cv2.waitKey(-1)