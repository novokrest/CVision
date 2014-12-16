__author__ = 'novokreshchenov.konstantin'

import cv2
import numpy as np


def make_fourier(src_img, wsize):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2

    fshift[crow - wsize: crow + wsize, ccol - wsize: ccol + wsize] = 0
    f_ishift = np.fft.ifftshift(fshift)
    f_img = np.abs(np.fft.ifft2(f_ishift))

    cv2.imwrite("fourier_w{0}.bmp".format(wsize), f_img)


def make_laplacian(src_img, kernel_size):
    l_img = cv2.Laplacian(src_img, cv2.CV_32F, ksize=kernel_size)
    cv2.imwrite("laplacian_ks{0}.bmp".format(kernel_size), l_img)


if __name__ == '__main__':
    img = cv2.imread('mandril.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)

    make_fourier(img, 20)
    make_fourier(img, 30)
    make_fourier(img, 40)

    make_laplacian(img, 1)
    make_laplacian(img, 3)