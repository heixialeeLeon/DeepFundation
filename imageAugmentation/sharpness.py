import numpy as np
from imageAugmentation.transform import Saturation
import cv2
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

def Sharpness_PIL(img, alpha):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(image)
    image_enhance = enhancer.enhance(alpha)
    image_enhance = np.array(image_enhance)
    image_enhance = image_enhance[:, :, ::-1]
    return image_enhance

def Sharpness_step_test(img):
    for alpha  in np.arange(0.5, 2.6, 0.5):
        cv2.imshow(f"{alpha}", Sharpness_PIL(img, alpha))
    cv2.waitKey(-1)

if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")
    cv2.imshow("raw", img)

    Sharpness_step_test(img)