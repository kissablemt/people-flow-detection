import cv2
import numpy as np

def scale(img, scale) -> np.ndarray:
    h, w = img.shape[:2]
    nw = scale * w
    nh = h / w * nw
    return cv2.resize(img, (int(nw), int(nh)))

def scale_by_width(img, new_width) -> np.ndarray:
    h, w = img.shape[:2]
    nw = new_width
    nh = int(h / w * nw)
    return cv2.resize(img, (nw, nh))

def scale_by_height(img, new_height) -> np.ndarray:
    h, w = img.shape[:2]
    nh = new_height
    nw = int(w / h * nh)
    return cv2.resize(img, (nw, nh))

def select_rectangle(img: np.ndarray, xyxy: list) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    ret = np.zeros(img.shape, np.uint8)
    ret[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return ret