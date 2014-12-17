__author__ = 'novokreshchenov.konstantin'

import cv2
import numpy as np

SOURCE_VIDEO_FILE_NAME = 'source.mpg'
HARRIS_VIDEO_FILE_NAME = 'harris.mpg'
FAST_VIDEO_FILE_NAME = 'fast.mpg'

MAX_TRACK_POINTS = 100


def harris_feature_detect(gray_img):
    HARRIS_QUALITY_LEVEL = 0.13
    HARRIS_MIN_DISTANCE = 5
    HARRIS_BLOCK_SIZE = 5

    h_features = cv2.goodFeaturesToTrack(gray_img,
                                         mask=None,
                                         maxCorners=MAX_TRACK_POINTS,
                                         qualityLevel=HARRIS_QUALITY_LEVEL,
                                         minDistance=HARRIS_MIN_DISTANCE,
                                         blockSize=HARRIS_BLOCK_SIZE,
                                         useHarrisDetector=True)
    return h_features


def fast_feature_detect(gray_img):
    fast = cv2.FastFeatureDetector()
    f_features = list(sorted(fast.detect(gray_img, None), key=lambda l: -l.response))[: MAX_TRACK_POINTS]

    return np.array([[f.pt] for f in f_features], np.float32)


def init_output_video_from_source(output_video_file, source_video):
    width, height = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), \
                    int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    result_video = cv2.VideoWriter(output_video_file,
                                   fourcc=cv2.cv.FOURCC('m', 'p', '4', 'v'),
                                   # fourcc=int(source_video.get(cv2.cv.CV_CAP_PROP_FOURCC)),
                                   fps=int(source_video.get(cv2.cv.CV_CAP_PROP_FPS)),
                                   frameSize=(width, height))
    return result_video


def read_video_gray_frame(video):
    status, frame = video.read()
    return status, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def process_optical_flow(source_video, output_video, features_getter):
    status, frame = source_video.read()
    if not status:
        return
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_features = features_getter(prev_frame)

    out_img = np.zeros_like(frame)
    colors = np.random.randint(0, 255, (MAX_TRACK_POINTS, 3))

    while True:
        status, frame = source_video.read()
        if not status:
            return

        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_features, None,
                                                            winSize=(15, 15),
                                                            maxLevel=2,
                                                            criteria=(
                                                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                10,
                                                                0.03))

        prev_points, next_points = prev_features[status == 1], next_features[status == 1]

        for i, (next, prev) in enumerate(zip(next_points, prev_points)):
            a, b = next.ravel()
            c, d = prev.ravel()
            cv2.line(out_img, (a, b), (c, d), colors[i].tolist(), 1)
            cv2.circle(frame, (a, b), 2, colors[i].tolist(), 2)

        output_frame = cv2.add(frame, out_img)
        output_video.write(output_frame)

        prev_frame = next_frame.copy()
        prev_features = next_points.reshape(-1, 1, 2)


if __name__ == '__main__':
    source_video = cv2.VideoCapture(SOURCE_VIDEO_FILE_NAME)
    harris_video = init_output_video_from_source(HARRIS_VIDEO_FILE_NAME, source_video)
    process_optical_flow(source_video, harris_video, harris_feature_detect)
    harris_video.release()

    source_video.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)
    fast_video = init_output_video_from_source(FAST_VIDEO_FILE_NAME, source_video)
    process_optical_flow(source_video, fast_video, fast_feature_detect)
    fast_video.release()

    source_video.release()
