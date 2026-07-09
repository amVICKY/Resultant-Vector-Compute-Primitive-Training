"""
evaluation/
-----------
Task-level evaluation pipelines used to relate weight-matrix *energy* (from the
Phase-1 rank/energy audit) to real model performance.

Two tasks are supported so that both metric families are covered:
  - classification  -> top-1 accuracy, macro precision   (Imagenette / pretrained ResNet)
  - segmentation    -> mean IoU, pixel accuracy, precision (Pascal VOC / pretrained FCN)
"""
