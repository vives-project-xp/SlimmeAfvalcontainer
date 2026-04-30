import time
import RPi.GPIO as GPIO

TRIG = 17
ECHO = 27

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
time.sleep(0.5) 

def measure():
    t = time.time()
    while GPIO.input(ECHO) == 1:
        if time.time() - t > 0.038:
            return None

    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG, GPIO.LOW)

    t = time.time()
    while GPIO.input(ECHO) == 0:
        if time.time() - t > 0.038:
            return None
    start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() - start > 0.038:
            return None
    stop = time.time()

    return ((stop - start) * 34300) / 2

print(f"Meten op TRIG=GPIO{TRIG}, ECHO=GPIO{ECHO} — Ctrl+C om te stoppen\n")

try:
    while True:
        dist = measure()
        if dist is None:
            print("Geen echo ontvangen")
        elif dist > 400:
            print("Buiten bereik")
        else:
            print(f"{dist:.1f} cm")
        time.sleep(0.5)
finally:
    GPIO.cleanup()
