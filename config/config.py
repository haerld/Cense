import torch

DetectionModel = 'models/yolo/yolo.onnx'
ClassificationModel1 = 'models/InceptionResnet/inception_resnet_model.pt'
ClassificationModel2 = 'models/ResNet/resnet_model.pt'
Classes = 9
Groups = ['00-10', '11-20', '21-30', 
          '31-40', '41-50', '51-60', 
          '61-70', '71-80', '81-90']

runOn = "cpu"