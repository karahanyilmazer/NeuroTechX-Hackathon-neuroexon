from exoskeleton_control.servo_control import ExoskeletonControl, GPIO
import numpy as np
import flask
from flask import Flask,render_template,Response,request
import time
import threading
import os
import json
from time import sleep
app = Flask(__name__)

def move_exoskeleton(ec):
    done = True
    while True:
        with open('./data.json','r') as f:
            try:
                detections = json.load(f)
            except:
                detections = {"move_command": [],
                              "angle": []
                              }
            
        move_command = detections['move_command']
        angle = detections['angle']
        
        if move_command and move_command[0]==True and done == True:
            if 0 <= angle[0] <= 180:
                done = False
                ec.SetAngle(angle[0])
                #sleep(1.5)
                request_data = {"move_command": [],
                                  "angle": []
                                  }
                with open('./data.json','w') as f:
                    json.dump(request_data,f)
                done = True

    ec.pwm.stop()
    GPIO.cleanup()
        

#to get json detection data in same port
@app.route('/detections', methods = ['POST'])
def detections():
    try:
        request_data = request.get_json()
        with open('data.json','w') as f:
            json.dump(request_data,f)
        return Response(print(request_data))
    except:
        return Response(print('[INFO]:Failed to update data.json'))

if __name__=="__main__":

    ec = ExoskeletonControl()
    threading.Thread(target=lambda: app.run(host='0.0.0.0',port=5000,debug=False)).start()
    move_exoskeleton(ec)
