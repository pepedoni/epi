import cv2
import uuid
from torch.autograd import Variable
import numpy as np
from imageai.Detection import ObjectDetection
import os

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.imagemJpeg = None
        self.lastImage  = None
        self.detector   = None
        self.execution_path = os.getcwd()
        self.inProcess  = False
        self.lastTextOutput = None

    def __del__(self):
        self.video.release()
    
    def process_image(self):
        try:
            self.inProcess = True
            
            self.textOutput = "["

            if (self.detector is None):
                self.detector = ObjectDetection()
                self.detector.setModelTypeAsRetinaNet()
                self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
                self.detector.loadModel()

            detections = self.detector.detectObjectsFromImage(input_image="./image.jpg", output_image_path="./image_trated.jpg")

            for eachObject in detections:
                self.textOutput = self.textOutput + '"' + eachObject["name"] + '"' + ','

            self.textOutput = self.textOutput[:-1]
            if(len(self.textOutput) != 0): 
                self.textOutput = self.textOutput + "]"
            else: 
                self.textOutput = "[]"


            self.inProcess = False
            self.lastTextOutput = self.textOutput
            return self.textOutput
        except:
            self.inProcess = False
            self.lastTextOutput = "[]"
            self.textOutput = "[]"
    
    def get_frame(self):
        self.success, self.image = self.video.read()

        if ( self.inProcess is False ):
            self.lastImage = self.image
            cv2.imwrite("./image.jpg", self.lastImage)

        ret, imagemJpeg = cv2.imencode('.jpg', self.image)


        self.imagemJpeg = imagemJpeg.tobytes()

