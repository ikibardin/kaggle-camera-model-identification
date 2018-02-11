import numpy as np
import math
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToPILImage, RandomCrop, \
    RandomHorizontalFlip, ToTensor, Normalize, CenterCrop, Lambda
import cv2
import numbers
import random
from io import BytesIO
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def _crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def _center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return _crop(img, i, j, th, tw)


def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, "Please provide only two dimensions (h, w) for size."
    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(size,
                                                                         (h,
                                                                          w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = _center_crop(img, (crop_h, crop_w))
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


def _full_d4(img):
    res = []
    for f in range(-1, 3):
        imgf = img.copy()
        if f == 0 or f == 1:
            imgf = cv2.flip(imgf, f)
        elif f == 2:
            imgf = cv2.flip(imgf, 0)
            imgf = cv2.flip(imgf, 1)
        for a in [0, 90]:
            imga = imgf.copy()
            rows, cols, c = imga.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), a, 1)
            imga = cv2.warpAffine(imga, M, (cols, rows))
            res.append(imga)
    return res


def _broken_d4(img):
    res = []
    for f in range(2):
        imgf = img.copy()
        imgf = cv2.flip(imgf, f)
        for a in [0, 90, 180, 270]:
            imga = imgf.copy()
            rows, cols, c = imga.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), a, 1)
            imga = cv2.warpAffine(imga, M, (cols, rows))
            res.append(imga)
    return res


class OpenCVRandomD4(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = _opencv_random_horizontal_flip(img)
        img = _opencv_random_vertical_flip(img)
        img = _opencv_random_rotate(img)
        return img


class FivePatchCrop(object):
    def __init__(self):
        pass


def _unalt_crop_base(crop, norm_mean, norm_std):
    return Compose([
        ToPILImage(),
        crop,
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std)
    ])


def unalt_random_crop(crop_size, norm_mean, norm_std):
    return _unalt_crop_base(RandomCrop(crop_size), norm_mean, norm_std)


def unalt_center_crop(crop_size, norm_mean, norm_std):
    return _unalt_crop_base(CenterCrop(crop_size), norm_mean, norm_std)


def yolo_crops(img, size):
    rcrop = RandomCrop(size)
    crops = []
    for i in range(256):
        crops.append(rcrop(img))
    return crops


def tta_transform(crop=_full_d4):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        # transforms.ToPILImage(),
        crop,
        Lambda(lambda crops: torch.stack(
            [transforms.Normalize(mean=norm_mean,
                                  std=norm_std)(ToTensor()(crop))
             for crop in crops]))
    ])


def index_gen(shape, shift, size):
    start = 0 if not shift else size // 2
    for i in range(start, shape[0] - size - size // 2, size):
        for j in range(start, shape[1] - size - size // 2, size):
            yield i, j


def get_crops(img, shift, size):
    crops = []
    for ly, lx in index_gen(img.shape, False, size):
        crops.append(img[ly: ly + size, lx: lx + size])
        assert crops[len(crops) - 1].shape == (
            size, size, 3), '{} | {}'.format(ly, lx)
    if not shift:
        return crops
    for ly, lx in index_gen(img.shape, True, size):
        crops.append(img[ly: ly + size, lx: lx + size])
        assert crops[len(crops) - 1].shape == (
            size, size, 3), '{} | {}'.format(ly, lx)
    return crops


alpha = 0.7
beta = 4.
gamma = 0.01


def _quality_base(mean, std):
    return alpha * beta * (mean - mean ** 2) + (1 - alpha) * (1 - gamma ** std)


def get_quality(img):
    # img /= 255.
    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    return _quality_base(means, stds).mean()


def quality_crops_tta_32(img, crops_count):
    crops = get_crops(img, False, 32)
    qs = np.array([get_quality(crop / 255.) for crop in crops])
    indices = list(reversed(qs.argsort()))
    return [crops[i] for i in indices[:crops_count]]


def after_offline_224(norm_mean, norm_std):
    return Compose([ToTensor(),
                    Normalize(mean=norm_mean, std=norm_std)])


def _jpeg_compression(img, quality):
    img = Image.fromarray(img)
    out = BytesIO()
    # print(type(quality), quality)
    img.save(out, format='jpeg', quality=int(quality))
    img = Image.open(out)
    return np.array(img)


def _opencv_resize(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def _opencv_gamma_correction(img, gamma):
    img = img / 255.
    img = cv2.pow(img, gamma)
    return (img * 255).astype(np.uint8)


class RandomManipTransform(object):
    def __init__(self, crop_size, bonus):
        self._crop = OpenCVRandomCrop(crop_size)
        self._bonus = bonus

    def __call__(self, img):
        transform_type = np.random.choice([0, 1, 2])
        if transform_type == 0:
            img = self._crop(img)
            if self._bonus:
                qual = np.random.choice([70, 75, 80, 85, 90])
            else:
                qual = np.random.choice([70, 90])
            return _jpeg_compression(img, qual)
        elif transform_type == 1:
            if self._bonus:
                factor = np.random.choice([0.5, 0.6, 0.8, 1.3, 1.5, 1.7, 2.0])
            else:
                factor = np.random.choice([0.5, 0.8, 1.5, 2.0])
            img = _opencv_resize(img, factor)
            return self._crop(img)
        elif transform_type == 2:
            img = self._crop(img)
            if self._bonus:
                g = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            else:
                g = np.random.choice([0.8, 1.2])
            return _opencv_gamma_correction(img, g)


def crops_augm_transform(crop_size, bonus):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return Compose([
        RandomManipTransform(crop_size, bonus),
        OpenCVRandomD4(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


class AugmForDetection:
    def __init__(self, crop_size):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        self._post_manip = Compose([
            OpenCVCenterCrop(crop_size),
            # OpenCVRandomD4(),
            ToTensor(),
            Normalize(norm_mean, norm_std)
        ])

    def __call__(self, img, label):
        """
        if label == 0:
            img = _jpeg_compression(img, 70)
        elif label == 1:
            img = _jpeg_compression(img, 90)
        elif label == 2:
            img = _opencv_resize(img, 0.5)
        elif label == 3:
            img = _opencv_resize(img, 0.8)
        elif label == 4:
            img = _opencv_resize(img, 1.5)
        elif label == 5:
            img = _opencv_resize(img, 2.0)
        elif label == 6:
            img = _opencv_gamma_correction(img, 0.8)
        elif label == 7:
            img = _opencv_gamma_correction(img, 1.2)
        else:
            assert False, 'Unexpected label'
        """
        img = random_manipulation(img, manipulation=MANIPULATIONS[label])
        img = self._post_manip(img)
        return img


MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

import jpeg4py as jpeg


def random_manipulation(img, manipulation=None):
    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        # img = np.array(img) # FIXME
        assert len(img.shape) == 3, 'actual shape {}'.format(img.shape)
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma) * 255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded


def resize_transform(crop_size, resize_factor):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    crop = OpenCVRandomCrop(crop_size)
    return Compose([
        lambda img: crop(_opencv_resize(img, resize_factor)),
        OpenCVRandomD4(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


def crop_and_d4_transform(crop_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return Compose([
        OpenCVRandomCrop(crop_size),
        OpenCVRandomD4(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


def center_crop_only(crop_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return Compose([
        OpenCVCenterCrop(crop_size),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


def d4_transform():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return Compose([
        OpenCVRandomD4(),
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


def normalize_only():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return Compose([
        ToTensor(),
        Normalize(norm_mean, norm_std)
    ])


class FiveCropWithD4TTA(object):
    def __init__(self, crop_size):
        self._size = crop_size

    def __call__(self, img):
        res = []
        crops = five_crop(img, self._size)
        for crop in crops:
            res += _full_d4(np.array(crop))
        return res


def five_crop_with_d4_tta(crop_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        OpenCVCenterCrop(crop_size),
        ToPILImage(),
        FiveCropWithD4TTA(crop_size),
        Lambda(lambda crops: torch.stack(
            [transforms.Normalize(mean=norm_mean,
                                  std=norm_std)(ToTensor()(crop))
             for crop in crops]))
    ])


class PseudoManipTransform:
    def __init__(self):
        self._random_d4 = d4_transform()
        self._crop_augm = crop_and_d4_transform(512)

    def __call__(self, img):
        if img.shape == (512, 512):
            return self._random_d4(img)
        else:
            return self._crop_augm(img)
