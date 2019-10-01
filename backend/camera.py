import cv2
import uuid
from torch.autograd import Variable
import numpy as np
from customEpiDetection import CustomObjectDetection
import os

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(1)
        self.imagemJpeg = None
        self.lastImage  = None
        self.trated_image = None
        self.detector   = None
        self.execution_path = os.getcwd()
        self.inProcess  = False
        self.lastTextOutput = None
        self.first = False

    def __del__(self):
        self.video.release()


    def process_image(self, file_image):
        try:
            self.inProcess = True
            
            self.textOutput = "["
            if (self.detector is None):
                self.detector = CustomObjectDetection()
                self.detector.setModelTypeAsYOLOv3()
                self.detector.setModelPath( os.path.join(self.execution_path , "epi.h5"))
                self.detector.setJsonPath("epi_config.json") 
                self.detector.loadModel()
                
                if(file_image == False):
                    self.trated_image, detections = self.detector.detectObjectsFromImage(input_type="array", input_image=self.lastImage, output_type="array", minimum_percentage_probability=80, display_percentage_probability=False)
                else:
                    detections = self.detector.detectObjectsFromImage(input_image="./image.jpg", output_image_path="./image_treated.jpg", minimum_percentage_probability=80, display_percentage_probability=False, thread_safe=True)

                self.detector.setEquipments(["pessoa", "capacete", "luva", "touca", "mascara"])

            else:
                if(file_image == False):
                    self.trated_image, detections = self.detector.detectObjectsFromImage(input_type="array", input_image=self.lastImage, output_type="array", minimum_percentage_probability=80, display_percentage_probability=False)
                else:
                    detections = self.detector.detectObjectsFromImage(input_image="./image.jpg", output_image_path="./image_treated.jpg", minimum_percentage_probability=80, display_percentage_probability=False, thread_safe=True)

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
        except Exception as e:
            self.inProcess = False
            self.lastTextOutput = None
            self.textOutput = "[]"
            raise e
    
    def get_frame(self, process_image):
        self.success, self.image = self.video.read()

        if ( process_image is True ):
            if ( self.inProcess is False ):
                self.lastImage = self.image
                outPut = self.process_image(False)
            ret, imagemJpeg = cv2.imencode('.jpg', self.trated_image)

        else:
            if ( self.inProcess is False ):
                self.lastImage = self.image
                cv2.imwrite("./image.jpg", self.lastImage)
            ret, imagemJpeg = cv2.imencode('.jpg', self.image)

        self.imagemJpeg = imagemJpeg.tobytes()

    def setEquipments(self, equipments):
        self.detector.setEquipments(equipments)

