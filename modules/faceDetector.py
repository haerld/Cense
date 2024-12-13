import config.config as config
import torch

FaceDetector = torch.hub.load('ultralytics/yolov5',
                              'custom',
                              config.DetectionModel,
                              _verbose=False)
FaceDetector.eval()
FaceDetector.to(config.runOn)