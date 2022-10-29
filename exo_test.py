from exoskeleton_control.servo_control import ExoskeletonControl, GPIO
from time import sleep
sc = ExoskeletonControl()
angles = [90,0,90,180]
for angle in angles:
    sc.SetAngle(angle)
    sleep(1.5)
    
sc.pwm.stop()
    
GPIO.cleanup()