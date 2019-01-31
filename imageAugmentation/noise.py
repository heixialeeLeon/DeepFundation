import numpy as np
from numpy import random
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageFilter
import skimage

def RandomNoise(img, noise):
    img = img.astype(np.float32)
    img = img / 255.0
    mode = [
        lambda x : skimage.util.random_noise(x, 'gaussian', mean=0, var=noise),
        lambda x : skimage.util.random_noise(x, 'speckle', mean=0, var=noise),
        lambda x : skimage.util.random_noise(x, 's&p', amount= noise)
    ]
    img = (random.choice(mode)(img)*255).astype(np.uint8)
    return img

if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")

    cv2.imshow("raw", img)

    img_blur = RandomNoise(img, 0.01)
    cv2.imshow("cv2", img_blur)
    cv2.waitKey(-1)