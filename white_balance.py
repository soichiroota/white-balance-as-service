import os
import io

import cv2 as cv
import numpy as np
from PIL import Image


def stretch_to_8bit(arr, clip_percentile=2.5):
    arr = np.clip(
        arr * (
            255.0 / np.percentile(arr, 100 - clip_percentile)
        ),
        0,
        255
    )
    return arr.astype(np.uint8)


def load_img(bytes_):
    bytes_io = io.BytesIO(bytes_)
    pil_img = Image.open(bytes_io)
    img_array = np.asarray(pil_img)
    bgr_img_array = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)
    return stretch_to_8bit(bgr_img_array)


def save_img(img_array, format_='PNG'):
    rgba_img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGBA)
    image = Image.fromarray(
        stretch_to_8bit(rgba_img_array),
        'RGBA'
    )
    with io.BytesIO() as bytes_io:
        image.save(bytes_io, format=format_)
        return bytes_io.getvalue()


class WhiteBalancer:
    def __init__(
        self,
        algo=None,
        range_thresh=255,
        bin_num=64,
        model_folder=None
    ):
        self.algo = algo
        self.range_thresh = range_thresh
        self.bin_num = 256 if range_thresh > 255 else 64
        self.model_path = ""
        if len(self.algo.split(":")) > 1:
            self.model_path = os.path.join(
                model_folder,
                algo.split(":")[1]
            )
        if algo == "grayworld":
            self.balance = self._apply_grayworld_algo
        elif algo.split(":")[0] == "learning_based":
            self.balance = self._apply_learning_based_algo
        else:
            self.balance = self._apply_simple_algo

    def _apply_grayworld_algo(self, im):
        inst = cv.xphoto.createGrayworldWB()
        inst.setSaturationThreshold(0.95)
        return inst.balanceWhite(im)

    def _apply_learning_based_algo(self, im):        
        inst = cv.xphoto.createLearningBasedWB(self.model_path)
        inst.setRangeMaxVal(self.range_thresh)
        inst.setSaturationThreshold(0.98)
        inst.setHistBinNum(self.bin_num)
        return inst.balanceWhite(im)

    def _apply_simple_algo(self, im):
        inst = cv.xphoto.createSimpleWB()
        return inst.balanceWhite(im)