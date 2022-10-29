import json
import requests
import cv2
import numpy as np
import time

#replace url according to raspberry ip:
url_detections = 'http://192.168.163.205:5000/detections'
font = cv2.FONT_HERSHEY_PLAIN

while True:
    move = input('set move to true or false\n')
    angle = input('set angle to 0 or 180\n')
    data = {"move_command": [bool(move)],
            "angle": [int(angle)]
            }

    try:
        server_return = requests.post(url_detections,json=data)
        print('[INFO]: Detections posted.')
    except:
        print('not able to connect')
        break
    