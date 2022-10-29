from exoskeleton_control.servo_control import ExoskeletonControl
import numpy as np
import flask
from flask import Flask,render_template,Response,request
import time
import threading
import os
import json
app = Flask(__name__)
def move_exoskeleton(ec):
    while True:
        with open('/data.json','r') as f:
            try:
                detections = json.load(f)
            except:
                detections = {"move_command": [],
                              "angle": []
                              }
            
        move_command = detections['move_command']
        angle = detections['angle']
    
        if move_command and move_command[0]==True:
            ec.SetAngle(angle)
            sleep(2.5)
            ec.SetAngle(90)
            move_command = False

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

# ec = ServoControl()
# angles = [90,0,90,180]
# for angle in angles:
#     ec.SetAngle(angle)
#     sleep(1.5)
    
# ec.pwm.stop()
    
# GPIO.cleanup()