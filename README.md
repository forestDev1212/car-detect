# car-detection-sample
This is python sample to detect cars.

<img src="./figures/sample_output.png" width=320></img>

# Environment Requirement
- python 3.10
- onnxruntime 1.16.1
- opencv-python 4.8.1.78

# Usage
If you run following command, coordinates of bounding box will be exported by json format.
```
$ python car_detection/predict.py car_detection/sample_image.jpg
{'bboxes': [{'id': 0, 'xmin': 3024, 'ymin': 2055, 'xmax': 3081, 'ymax': 2119}, {'id': 1, 'xmin': 2592, 'ymin': 2039, 'xmax': 2689, 'ymax': 2079}, {'id': 2, 'xmin': 1958, 'ymin': 1984, 'xmax': 2335, 'ymax': 2163}, {'id': 3, 'xmin': 3406, 'ymin': 1626, 'xmax': 4032, 'ymax': 2889}], 'image_width': 4032, 'image_height': 3024}
```"# car-detect" 
"# car-detect" 
