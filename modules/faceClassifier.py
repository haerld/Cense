import torchvision.models as models
from torch import nn
import config.config as config
import torch

FaceClassifier = models.resnet18(pretrained=True)
FaceClassifier.to(config.runOn)

FaceClassifier.fc = nn.Linear(512, config.Classes+2)
FaceClassifier = nn.Sequential(FaceClassifier, nn.Sigmoid())

FaceClassifier.load_state_dict(torch.load(config.ClassificationModel2, map_location=torch.device('cpu')))

FaceClassifier.eval()