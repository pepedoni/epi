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
                                                    
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/process_image', methods=['GET'])
def process_image():
    response = app.response_class(
        response=json.dumps(genObjects()),
        status=200,
        mimetype='application/json'
    )
    return response

def genObjects():

    if( camera.inProcess is True):
        output = camera.lastTextOutput
        if(output is None):
            return "[]"
        else:
            return output
    else: 
        output = camera.process_image()
        return output

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    global camera
    
    # camera.processarCamera()

    while True:
        camera.get_frame(True)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + camera.imagemJpeg + b'\r\n\r\n')


app.run(host="0.0.0.0", debug=True)