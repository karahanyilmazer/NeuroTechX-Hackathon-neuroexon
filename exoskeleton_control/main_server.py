import requests

#replace url according to raspberry ip:
url_detections = 'http://192.168.163.205:5000/detections'

while True:
    angle = input('set angle to 0 or 180\n')
    data = {"move_command": [True],
            "angle": [int(angle)]
            }

    try:
        server_return = requests.post(url_detections,json=data)
        print('[INFO]: Detections posted.')
    except:
        print('not able to connect')
        break
    