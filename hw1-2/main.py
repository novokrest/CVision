__author__ = 'novokonst'

import cv2
import numpy as np
import io_settings as io

IMG_PATH = 'text.bmp'
SETTINGS_PATH = 'settings.txt'


def dilate(in_img, params):
    kernel = np.ones((params['dilate_kernelX'], params['dilate_kernelY']), np.uint8)
    out_img = cv2.dilate(in_img, kernel, iterations=1)
    return out_img


def erode(in_img, params):
    kernel = np.ones((params['erode_kernelX'], params['erode_kernelY']), np.uint8)
    out_img = cv2.erode(in_img, kernel, iterations=1)
    return out_img


def draw_contours(img):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(cv2.convertScaAbs(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img


def draw_binary_components(in_img):
    out_img = cv2.threshold(in_img, 128, 255, cv2.THRESH_BINARY)[1]
    img_height, img_width = out_img.shape
    mask = np.zeros((img_height + 2, img_width + 2), np.uint8)
    rectangles = []
    for i in range(img_width):
        for j in range(img_height):
            if out_img[j, i] != 255:
                continue
            res, rectangle = cv2.floodFill(out_img, mask, (i, j), 255, flags=cv2.FLOODFILL_MASK_ONLY)
            if res:
                rectangles.append(rectangle)

    out_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR)
    for rectangle in rectangles:
        x, y, w, h = rectangle
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return out_img


if __name__ == '__main__':
    settings = io.read_settings(SETTINGS_PATH)

    # HW 1
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    th_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

    g_img = cv2.GaussianBlur(src=th_img,
                             ksize=(settings['gaussian_ksizeX'], settings['gaussian_ksizeY']),
                             sigmaX=settings['gaussian_sigmaX'])

    l_img = cv2.Laplacian(src=g_img,
                                ddepth=cv2.CV_32F,
                                ksize=settings['laplacian_ksize'],
                                scale=settings['laplacian_scale'])
    cv2.imwrite('Gauss&Laplace.bmp', l_img)

    result1_img = draw_binary_components(l_img)
    cv2.imwrite('hw1_result.bmp', result1_img)

    # HW 2
    img = 255 - cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    img = dilate(img, {'dilate_kernelX': 3, 'dilate_kernelY': 5})
    # cv2.imwrite('dilate1.bmp', img)
    img = erode(img, {'erode_kernelX': 3, 'erode_kernelY': 3})
    # cv2.imwrite('erode1.bmp', img)
    img = dilate(img, {'dilate_kernelX': 3, 'dilate_kernelY': 1})
    # cv2.imwrite('dilate2.bmp', img)
    img = erode(img, {'erode_kernelX': 3, 'erode_kernelY': 3})
    # cv2.imwrite('erode2.bmp', img)

    result_img = draw_binary_components(img)
    cv2.imwrite('hw2_result.bmp', result_img)