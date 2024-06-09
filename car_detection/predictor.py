from pathlib import Path

import numpy as np
import onnxruntime
import cv2


def normalize(img, mean, std, max_pixel_value=255.0):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def normalize_01(img, min=None, max=None, dtype=np.float32):
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    if max == min:
        out = np.zeros_like(img)
        if dtype is not None:
            out = out.astype(dtype)
        return out
    out = (img - min) / (max - min)
    if dtype is not None:
        out = out.astype(dtype)
    return out


class Predictor(object):
    def __init__(
        self,
        model_path: str,
        is_gpu: bool = False,
        cpus: int = 4,
        model_output_ch=None,
        normalize_type="normalize",
        *args,
        **kwargs,
    ):
        """
        Class to run onnx model for segmentation.

        model_path: path to ONNX model path.

        is_gpu: If True, use GPU.

        model_output_ch: the ONNX model output label when the model is 1 class model
        """
        self.model_path = model_path
        self.is_gpu = is_gpu
        self.cpus = cpus
        self.model_output_ch = model_output_ch
        self.normalize_type = normalize_type

        self.load_model()

    def load_model(self):
        if self.is_gpu:
            device = ["CUDAExecutionProvider"]
        else:
            device = ["CPUExecutionProvider"]
        print("Loading", self.model_path)
        self.sess = onnxruntime.InferenceSession(self.model_path, providers=device)
        # input_shape : (Batch, ch, h, w)
        input_shape = self.sess.get_inputs()[0].shape
        self.input_size = (input_shape[-2], input_shape[-1])

    def load_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, str):
            return input_tensor
        if not Path(input_tensor).is_file():
            raise FileNotFoundError(f"No file with path {input_tensor} exists!")
        raise NotImplementedError()

    def preprocess(self, input_img, normalize_type):
        """
        preprocess input image
        1. normalize: specifiy the same normalize function when training.
        2. resize
        3. convert to (1, CH, H, W)

        input_img: numpy.ndarray image

        normalize_type: normalize type

        """
        if normalize_type in ["normalize"]:
            img = normalize(
                input_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        elif normalize_type in ["normalize_01"]:
            img = normalize_01(input_img, min=0, max=255, dtype=np.float32)
        elif normalize_type in ["normalize_01_float"]:
            img = normalize_01(input_img, min=None, max=None, dtype=np.float32)
        else:
            raise NotImplementedError(
                "undefined normalize type {}".format(normalize_type)
            )

        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        # H x W x CH image  --->   N x CH x H x W
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, prediction, shape=None):
        """
        postprocess to get segmemtation mask with the same size as input image.
        """
        # outs list of ndarray. ndarray dims: batch, nclass, height, width.
        # 1 x CH x H x W    ->    H x W x CH
        mask = np.moveaxis(prediction[0][0], 0, -1)
        # mask = np.rollaxis(prediction[0][0])

        if shape is None:
            mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]))
        else:
            mask = cv2.resize(mask, (shape[1], shape[0]))
        return mask

    def infer(self, input_img):
        """
        run inference
        1. preprocess input image
        2. run onnxruntime session
        3. postprocess
          suppose segmentation model by default.

        input_img: numpy.ndarray
        """
        # input_tensor = self.load_input_tensor(input_tensor)
        shape = input_img.shape
        input_img = self.preprocess(input_img, self.normalize_type)

        if len(self.sess.get_inputs()) > 1:
            raise NotImplementedError()

        inputs = {self.sess.get_inputs()[0].name: input_img}

        outs = self.sess.run(None, inputs)

        mask = self.postprocess(outs, shape)

        return mask

    def infer_on_batch(self, input_imgs):
        raise NotImplementedError()
