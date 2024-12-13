from PIL import ImageDraw, Image
from modules import faceDetector, faceClassifier
import time
from core import utils as utils


def returnAnalysis(imagePath):
    IMG = Image.open(imagePath)
    IMG_ = ImageDraw.Draw(IMG)
    t0 = time.time()
    faces, bbs = utils.extractFace(IMG, faceDetector.FaceDetector, 0.7)
    tt1 = time.time() - t0

    tt2 = 0
    for face, bb in zip(faces, bbs):
        IMG_.rectangle(bb, outline ="Red", width=2)
        tensorIMG = utils.readImage(face)
        t0 = time.time()
        Age, Gender, C = utils.extractInfo(faceClassifier.FaceClassifier, tensorIMG)
        tt = time.time() - t0
        tt2 += tt
        textBox = f'{Age}'
        Text = ImageDraw.Draw(IMG)
        Text.text((bb[0][0]+5, bb[0][1]+2),
                    textBox,
                    fill=(255, 0, 0))
    return IMG, tt1+tt2