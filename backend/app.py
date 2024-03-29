import sys
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from camera import VideoCamera
import cv2
import requests
import json

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

camera = VideoCamera()
imagemOriginal = None
treat_image = True
                                                    
@app.route('/process_image', methods=['GET'])
def process_image():
    response = app.response_class(
        response=json.dumps(genObjects()),
        status=200,
        mimetype='application/json'
    )
    return response

def genObjects():

    if( camera.inProcess is True ):
        output = camera.lastTextOutput
        if(output is None):
            return "[]"
        else:
            return output
    else: 
        if( treat_image == False ):
            output = camera.process_image(True)
        else:
            return camera.lastTextOutput 
        return output

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    global camera

    while True:
        camera.get_frame(treat_image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + camera.imagemJpeg + b'\r\n\r\n')

@app.route('/set_equipments', methods = ['POST'])
def set_equipments():
    data = request.json
    camera.setEquipments(data["equipments"])
    response = app.response_class(
        response=json.dumps(["OK"]),
        status=200,
        mimetype='application/json'
    )
    return response


app.run(host="0.0.0.0", debug=True)