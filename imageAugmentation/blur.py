import numpy as np
from numpy import random
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageFilter

def RandomBlur(img, radius):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    filter = [
        ImageFilter.GaussianBlur(radius),
        ImageFilter.BoxBlur(radius),
        ImageFilter.MedianFilter(size=3)
    ]
    img = img.filter(random.choice(filter))
    img = np.array(img)
    img = img[:,:,::-1]
    return img

if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")
    cv2.imshow("raw", img)

    img_blur = RandomBlur(img, 3)
    cv2.imshow("cv2", img_blur)
    cv2.waitKey(-1)