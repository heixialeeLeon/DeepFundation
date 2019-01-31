import numpy as np
from imageAugmentation.transform import Saturation
import cv2
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from numpy import random

def Saturation_CV(img, alpha):
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img[:, :, 1] *= alpha
    img[:,:,1] = np.clip(img[:,:,1]*alpha,0,255)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = img.astype(np.uint8)
    return img

def Saturation_PIL(img, alpha):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(image)
    image_enhance = enhancer.enhance(alpha)
    image_enhance = np.array(image_enhance)
    image_enhance = image_enhance[:, :, ::-1]
    return image_enhance

def RandomSaturation_PIL(img, lower=0.5, upper=1.5):
    alpha = random.uniform(lower, upper)
    return Saturation_PIL(img, alpha)

def Saturation_step_test(img):
    for alpha  in np.arange(0.5, 2.6, 0.5):
        cv2.imshow(f"{alpha}", Saturation_PIL(img, alpha))
    cv2.waitKey(-1)

if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")
    cv2.imshow("raw", img)

    # img_cv = Saturation(img, 1.5)
    # img_pil = Saturation_PIL(img, 1.5)
    # cv2.imshow("cv2", img_cv)
    # cv2.imshow("pil", img_pil)
    # cv2.waitKey(-1)

    Saturation_step_test(img)