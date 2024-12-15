import config.config as config
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([transforms.ToTensor()])

def extractFace(IMG, FaceDetector, threshold=0.50, returnFace=True):
    extractedFaces = []
    extractedBoxes = []
    FaceDetections = FaceDetector(IMG).pandas().xyxy[0]
    for detection in FaceDetections.values:
        xmin, ymin, xmax, ymax, confidence = detection[:5]
        if confidence >= threshold:
            bb = [(xmin, ymin), (xmax, ymax)]
            if returnFace:
                w, h = xmax - xmin, ymax - ymin
                currentFace = IMG.crop((xmin, ymin, w+xmin, h+ymin))
                extractedFaces.append(currentFace)
            extractedBoxes.append(bb)

    return extractedFaces, extractedBoxes

def readImage(IMG):
    IMG = IMG.convert('RGB')
    IMG = IMG.resize((200, 200))
    tensorIMG = transform(IMG).unsqueeze(0)
    return tensorIMG

def extractInfo(MyModel, tensorIMG, Verbosity=False):
    tensorIMG = tensorIMG.to(config.runOn)
    tensorLabels = MyModel(tensorIMG)[0]

    Age = torch.argmax(tensorLabels[:config.Classes])

    C1 = float(torch.max(tensorLabels[:config.Classes]))
    C2 = float(torch.max(tensorLabels[config.Classes:]))

    if Verbosity:
        output = [round(float(x), 3) for x in tensorLabels]
        print(output)
    return config.Groups[Age], [round(C1, 3), round(C2, 3)]