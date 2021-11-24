# YOLOv3 with circular bounding box

This repository contains implementation of YOLOv3 from scratch for circular bounding boxes.

For circular bounding box, only 3 attributes are required:

- center x-coordinate
- center y-coordinate
- radius

To download YOLOv3 weights use the below command:

```
!wget https://pjreddie.com/media/files/yolov3.weights
```

## Modifications

- Rectangular anchor box has two attributes, width and height, but for circular anchor box only radius is required.

  - Update the `anchors` of yolo blocks in yolov3.cfg.
  - To maintain the size of the bboxes, the radius is kept equal to half the width of a rectangular anchor box.

- In `_detection()` method, transformation is applied to three attrributes instead of four like in rectangular bbox.

  - Apply sigmoid to x and y coordinates of the center and add the corresponding offset.
  - Scale the radius exponentially to match the size of the circular anchor box.

- For calculating the intersection area between two circles, I referred to [this](https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/) formula.

- In `bbox_detection()` method, top left and bottom right coordinates of bbox are no longer required, so I have commented out those lines.

For a better comparison, I have created different scripts and config files for rectangular and circular bboxes:

- Rectangle: `main.py` and `yolov3.cfg`
- Circle: `main_circle.py` and `yolov3_circle.cfg`

For this implementation I have referred to [this](https://medium.datadriveninvestor.com/yolov3-from-scratch-using-pytorch-part1-474b49f7c8ef) blog.
