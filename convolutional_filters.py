import numpy as np
import cv2


def apply_edge_detection(matrix):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(matrix, -1, kernel)


def apply_sharpening(matrix):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(matrix, -1, kernel)

    
def apply_box_blur(matrix, ker=3):
    kernel = (1./ker**2) * np.ones((ker, ker), float)
    return cv2.filter2D(matrix, -1, kernel)

    
def apply_Gaussian_blur(matrix):
    kernel = (1./16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return cv2.filter2D(matrix, -1, kernel)    