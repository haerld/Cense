from facenet_pytorch import InceptionResnetV1
from torch import nn
import config.config as config
import torch

FaceClassifier = InceptionResnetV1(pretrained='vggface2', classify=True)

FaceClassifier.logits = nn.Linear(512, config.Classes+2, bias=True)
FaceClassifier = nn.Sequential(FaceClassifier, nn.Dropout(0.5))
FaceClassifier = nn.Sequential(FaceClassifier, nn.Sigmoid())

FaceClassifier.load_state_dict(torch.load(config.ClassificationModel1, map_location=torch.device('cpu')))
FaceClassifier.eval()

FaceClassifier.to(config.runOn)