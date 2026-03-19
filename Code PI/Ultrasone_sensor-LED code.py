import RPi.GPIO as GPIO
import time

# GPIO Pin Configuration
SENSOR_1_TRIG = 17
SENSOR_1_ECHO = 27
SENSOR_2_TRIG = 22
SENSOR_2_ECHO = 4
SENSOR_3_TRIG = 24
SENSOR_3_ECHO = 25  
SENSOR_4_TRIG = 8   
SENSOR_4_ECHO = 7

CONTAINER_HEIGHT_CM = 65.0  # adjust to your container height
DEBOUNCE_TIME = 1.5

SENSORS = [
    {"id": 1, "name": "Restafval", "trig": SENSOR_1_TRIG, "echo": SENSOR_1_ECHO, "threshold": 10},
    # Sensor 2/3/4 tijdelijk uitgeschakeld.
    # Voeg ze later weer toe zodra deze GPIO's beschikbaar zijn.
]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for sensor in SENSORS:
    # Op Pi 5 (rpi-lgpio backend) kan setup zonder initial soms falen.
    GPIO.setup(sensor["trig"], GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(sensor["echo"], GPIO.IN)


def measure_distance(sensor, timeout=0.02):
    GPIO.output(sensor["trig"], GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(sensor["trig"], GPIO.LOW)

    start_time = time.time()
    stop_time = start_time

    # Fixed timeout logic
    pulse_start_timeout = time.time()
    while GPIO.input(sensor["echo"]) == 0:
        if time.time() - pulse_start_timeout >= timeout:
            return None
        start_time = time.time()

    pulse_stop_timeout = time.time()
    while GPIO.input(sensor["echo"]) == 1:
        if time.time() - pulse_stop_timeout >= timeout:
            return None
        stop_time = time.time()

    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2

    if distance <= 2 or distance > 400:
        return None

    return distance


def measure_average(sensor, samples=3):
    readings = []
    for _ in range(samples):
        d = measure_distance(sensor)
        if d is not None:
            readings.append(d)
        time.sleep(0.03)
    return sum(readings) / len(readings) if readings else None


def calc_fill_percentage(current_distance):
    if current_distance is None:
        return -1
    pct = ((CONTAINER_HEIGHT_CM - current_distance) / CONTAINER_HEIGHT_CM) * 100
    return max(0, min(100, int(pct)))


def check_garbage_disposed(sensor, previous_distance, current_distance):
    if current_distance is None or previous_distance is None:
        return False, previous_distance or current_distance

    delta = previous_distance - current_distance

    if delta >= sensor["threshold"]:
        time.sleep(0.8)  # wait for object to settle
        confirmed = measure_average(sensor, samples=5)
        if confirmed is not None:
            if previous_distance - confirmed >= sensor["threshold"]:
                return True, confirmed

    return False, current_distance


def check_container_full(current_distance):
    if current_distance is None:
        return False
    return current_distance < 5


def main():
    previous_distances = [measure_average(s, samples=5) for s in SENSORS]
    last_throw_time = [0.0] * len(SENSORS)

    print("Smart bin ready!\n")

    try:
        while True:
            for i, sensor in enumerate(SENSORS):
                now = time.time()

                # Debounce check
                if now - last_throw_time[i] < DEBOUNCE_TIME:
                    continue

                current_distance = measure_average(sensor, samples=3)

                if current_distance is None:
                    print(f"Sensor {sensor['id']} ({sensor['name']}): no echo.")
                    continue

                fill = calc_fill_percentage(current_distance)
                print(
                    f"Sensor {sensor['id']} ({sensor['name']}): "
                    f"afstand={current_distance:.1f}cm | vulling={fill}%"
                )
                disposed, updated = check_garbage_disposed(sensor, previous_distances[i], current_distance)

                if disposed:
                    last_throw_time[i] = time.time()
                    print(f"✅ Garbage disposed in {sensor['name']} "
                          f"({previous_distances[i]:.1f}cm → {updated:.1f}cm) | Fill: {fill}%")

                if check_container_full(current_distance):
                    print(f"⚠️  {sensor['name']} is FULL ({current_distance:.1f}cm)!")

                previous_distances[i] = updated
                time.sleep(0.05)

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()