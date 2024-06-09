import argparse

import cv2
import numpy as np

from predictor import Predictor
from util import segmentation_mask_to_bounding_boxes, bbox2dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="path to image")
    parser.add_argument(
        "--model",
        type=str,
        help="path to onnx model",
        default="./car_detection/sample-model.onnx",
    )
    args = parser.parse_args()

    model = Predictor(args.model)

    input = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    seg = model.infer(input)

    bboxes = segmentation_mask_to_bounding_boxes(np.uint8(seg > 0.5))

    output = bbox2dict(bboxes)
    output["image_width"] = input.shape[1]
    output["image_height"] = input.shape[0]
    print(output)


if __name__ == "__main__":
    main()
