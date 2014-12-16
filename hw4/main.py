__author__ = 'novokrest'

import cv2
import numpy as np

SOURCE_IMAGE_FILE           = 'mandril.bmp'
RR_IMAGE_FILE               = 'rr_mandril.bmp'
SOURCE_KEYPOINTS_IMAGE_FILE = 'keypoints.bmp'
RR_KEYPOINTS_IMAGE_FILE     = 'rr_keypoints.bmp'
RESULT_IMAGE_FILE           = 'result.bmp'


def resize_and_rotate(src_img, scale, angle):
    src_height, src_width = src_img.shape[:2]
    rr_matr = cv2.getRotationMatrix2D((src_width / 2, src_height / 2), angle, scale)
    rr_img = cv2.warpAffine(src_img, rr_matr, (src_width, src_height))

    return rr_img, rr_matr


def get_matches(desc_src, desc_rr):
    index_params, search_params = dict(algorithm=0, trees=5), dict(checks=50)
    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(desc_src, desc_rr, k=2)

    matchesMask = [0 for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = 1

    return [match for (i, match) in enumerate(matches) if matchesMask[i] == 1]


def draw_matches(img1, kp1, img2, kp2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    out_img = np.zeros((max(rows1, rows2), cols1 + cols2, 3), np.uint8)
    out_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    out_img[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    for match in matches:
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out_img, (int(x1), int(y1)), 4, (0, 0, 255), 2)
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 4, (0, 0, 255), 1)

        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)

    return out_img


if __name__ == '__main__':
    src_img = cv2.imread(SOURCE_IMAGE_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    rr_img, rr_matrix = resize_and_rotate(src_img, 0.5, 45)

    sift = cv2.SIFT(nfeatures=1000) #, nOctaveLayers=4, contrastThreshold=0.03, edgeThreshold=10)
    kp_src, desc_src = sift.detectAndCompute(src_img, None)
    kp_rr, desc_rr = sift.detectAndCompute(rr_img, None)

    matches = get_matches(desc_src, desc_rr)
    all_matches_count = len(matches)
    real_matches_count = 0
    for match in matches:
        p = np.append(kp_src[match[0].queryIdx].pt, 1)
        real_rr_point = np.array([
            np.dot(p, rr_matrix[0, :]),
            np.dot(p, rr_matrix[1, :])
        ])
        error = np.linalg.norm(np.array(real_rr_point) - kp_rr[match[0].trainIdx].pt)
        if error < 0.75:
            real_matches_count += 1

    match_precision = 1.0 * real_matches_count / all_matches_count
    report = "{0} points from {1} points were matched correctly. Matching precision equals {2}%"\
        .format(real_matches_count, all_matches_count, match_precision)
    print(report)

    # save results
    kp_src_img = cv2.drawKeypoints(src_img, kp_src, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_rr_img = cv2.drawKeypoints(rr_img, kp_rr, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_img = draw_matches(src_img, kp_src, rr_img, kp_rr, matches)

    cv2.imwrite(SOURCE_KEYPOINTS_IMAGE_FILE, kp_src_img)
    cv2.imwrite(RR_IMAGE_FILE, rr_img)
    cv2.imwrite(RR_KEYPOINTS_IMAGE_FILE, kp_rr_img)
    cv2.imwrite(RESULT_IMAGE_FILE, result_img)