import cv2


def segmentation_mask_to_bounding_boxes(binary_mask):
    """
    Calculate the bounding boxes of the objects in the binary segmentation mask.

    Parameters:
    - binary_mask: 2D numpy array representing the binary mask

    Returns:
    - list of bounding boxes [(xmin1, ymin1, xmax1, ymax1), ...]
    """

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = [(x, y, x + w, y + h) for x, y, w, h in bounding_boxes]

    return bounding_boxes


def draw_bounding_boxes_on_image(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw multiple bounding boxes on the input image.

    Parameters:
    - image: input image
    - bboxes: list of bounding boxes
    - color: tuple representing the RGB values for the bounding box color
    - thickness: thickness of the bounding box

    Returns:
    - image with bounding boxes drawn on it
    """

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

    return image


def bbox2dict(bboxes):
    output = []
    for i, b in enumerate(bboxes):
        output.append({"id": i, "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    return {"bboxes": output}
