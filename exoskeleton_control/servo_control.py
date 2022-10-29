import RPi.GPIO as GPIO
from time import sleep

class ExoskeletonControl():
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(11,GPIO.OUT)
        GPIO.setup(13,GPIO.OUT)
        self.pwm = GPIO.PWM(11,50)
        self.pwm.start(0)
        
    def SetAngle(self,angle):
        self.give_feedback()
        duty = angle / 18 + 2
        GPIO.output(11, True)
        self.pwm.ChangeDutyCycle(duty)
        sleep(1)
        GPIO.output(11, False)
        self.pwm.ChangeDutyCycle(0)

    def give_feedback(self):
        GPIO.output(13, True)
        sleep(0.5)
        GPIO.output(13, False)

