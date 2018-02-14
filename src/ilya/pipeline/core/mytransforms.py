import numpy as np
import cv2


def five_crop(img, size):
    h, w, c = img.shape
    assert c == 3, 'Something wrong with channels order'
    if size > w or size > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(size, (h, w)))
    tl = img[0: size, 0: size]
    tr = img[0: size, w - size: w]
    bl = img[h - size: h, 0: size]
    br = img[h - size: h, w - size: w]
    center = OpenCVCenterCrop(size)(img)
    return tl, tr, bl, br, center


class OpenCVCropBase(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, img):
        i, j = self.get_params(img)
        return img[i: i + self._size, j: j + self._size]


class OpenCVCenterCrop(OpenCVCropBase):
    def __init__(self, size):
        super().__init__(size)

    def get_params(self, img):
        h, w, c = img.shape
        if h == self._size and w == self._size:
            return 0, 0
        th = tw = self._size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j


class OpenCVRandomCrop(OpenCVCropBase):
    def __init__(self, size):
        super().__init__(size)

    def get_params(self, img):
        h, w, c = img.shape
        if h == self._size and w == self._size:
            return 0, 0
        if 0 >= h - self._size or 0 >= w - self._size:
            print(h, w)
        i = 0 if h == self._size else np.random.randint(0, h - self._size)
        j = 0 if w == self._size else np.random.randint(0, w - self._size)
        return i, j


def _opencv_random_horizontal_flip(img):
    if np.random.random() > 0.5:
        return img
    return cv2.flip(img, 1)


def _opencv_random_vertical_flip(img):
    if np.random.random() > 0.5:
        return img
    return cv2.flip(img, 0)


def _opencv_random_rotate(img):
    rows, cols, c = img.shape
    angle = np.random.choice([0, 90, 180, 270])
    if angle == 0:
        return img
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


class OpenCVRandomD4(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = _opencv_random_horizontal_flip(img)
        img = _opencv_random_vertical_flip(img)
        img = _opencv_random_rotate(img)
        return img
