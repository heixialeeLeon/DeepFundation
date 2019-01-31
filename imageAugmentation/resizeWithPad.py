import cv2
import math
import numpy as np
from numpy import random

def ResizeWithPad_W(img, target_size, new_value=[127,127,127]):
    ih,iw,_ = img.shape
    th,tw = target_size
    resize_rate = float(th/ih)
    resize_w = int(round(iw*resize_rate))
    resize_h = int(round(ih*resize_rate))
    resize_img = cv2.resize(img, (resize_w,resize_h))

    new_img = np.full((th,tw,3), new_value, dtype=np.uint8)

    start_w =random.randint(0, max(0, tw-resize_w))
    new_img[:resize_h,start_w:start_w+resize_w,:] = resize_img
    return new_img

def ResizeWithPad_H(img, target_size, new_value=[127,127,127]):
    ih, iw, _ = img.shape
    th, tw = target_size
    resize_rate = float(tw / iw)
    resize_w = int(round(iw * resize_rate))
    resize_h = int(round(ih * resize_rate))
    resize_img = cv2.resize(img, (resize_w, resize_h))

    new_value = [127, 127, 127]
    new_img = np.full((th, tw, 3), new_value, dtype=np.uint8)

    start_h = random.randint(0, max(0, th - resize_h))
    new_img[start_h:start_h+resize_h, :resize_w, :] = resize_img
    return new_img


if __name__ == "__main__":
    img = cv2.imread("../test/1.jpg")
    cv2.imshow("raw", img)

    img_resize_w = ResizeWithPad_W(img, (200, 400), [255,0,0])
    img_resize_h = ResizeWithPad_H(img, (400, 400))
    cv2.imshow("resize_w", img_resize_w)
    cv2.imshow("resize_h", img_resize_h)
    cv2.waitKey(-1)