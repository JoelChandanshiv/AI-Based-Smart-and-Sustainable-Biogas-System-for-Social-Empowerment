# inference_sorter.py
import time
import os
import cv2
import numpy as np
import json
from datetime import datetime
import threading

# For TFLite: try tflite_runtime, fallback to tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter

# GPIO libs (RPi)
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None
    print("RPi.GPIO not available on this machine. Running in emulation mode.")

# Optional MQTT
try:
    import paho.mqtt.publish as publish
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False

# CONFIG - customize to your hardware
MODEL_PATH = "model_int8.tflite"
IMG_SIZE = 224
CAMERA_INDEX = 0  # /dev/video0 or PiCamera config
CONF_ACCEPT = 0.80   # probability threshold to auto-accept
CONF_REJECT = 0.50   # below this auto-reject; between -> audit
SNAPSHOT_DIR = "snapshots"
LOG_FILE = "classification_log.csv"

# GPIO pins - update to match your wiring
ENCODER_PIN = 17     # encoder signal (BCM)
SERVO_PIN = 18       # servo PWM pin (BCM)
CONVEYOR_ENABLE_PIN = 27  # optional motor enable

# Actuator timing: distance from camera to pusher (m) and conveyor speed (m/s)
DISTANCE_M = 0.30    # physical distance between camera FOV line and actuator
CONVEYOR_SPEED_MPS = 0.10  # default slow speed (m/s)
# If you have encoder ticks per meter, set it:
TICKS_PER_METER = 200.0   # encoder counts per meter (calibrate)
TICKS_TO_ACTUATOR = int(TICKS_PER_METER * DISTANCE_M)

# Encoder counters
encoder_count = 0
encoder_lock = threading.Lock()

# Setup directories
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,frame_id,label,prob,decision,encoder_tick,image\n")

# GPIO setup
def gpio_setup():
    if GPIO is None:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ENCODER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    GPIO.setup(CONVEYOR_ENABLE_PIN, GPIO.OUT)
    # Start conveyor (optional)
    try:
        GPIO.output(CONVEYOR_ENABLE_PIN, GPIO.HIGH)
    except Exception:
        pass

# Encoder callback
def encoder_callback(channel):
    global encoder_count
    with encoder_lock:
        encoder_count += 1

# Initialize encoder interrupt
def start_encoder_listener():
    if GPIO is None:
        return
    GPIO.add_event_detect(ENCODER_PIN, GPIO.RISING, callback=encoder_callback, bouncetime=1)

# Servo control - simple PWM-based push
class Servo:
    def __init__(self, pin):
        self.pin = pin
        self.pwm = None
        if GPIO is not None:
            self.pwm = GPIO.PWM(pin, 50)  # 50Hz
            self.pwm.start(0)
        # servo positions (duty cycles) -- calibrate per hardware
        self.rest_duty = 7.5
        self.push_duty = 12.0
        self.push_time = 0.25  # seconds to hold push

    def push(self):
        if GPIO is None:
            print("[SIM] servo push")
            return
        self.pwm.ChangeDutyCycle(self.push_duty)
        time.sleep(self.push_time)
        self.pwm.ChangeDutyCycle(self.rest_duty)
        time.sleep(0.05)
        self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        if self.pwm:
            self.pwm.stop()

# TFLite helper (supports uint8 quantized model)
class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]["dtype"]
        print("TFLite input dtype:", self.input_dtype)

    def preprocess(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(img, 0)
        if self.input_dtype == np.uint8:
            # If model uses uint8, use same scale as during conversion
            # Many quantized models expect 0..255 uint8 input
            input_data = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            input_data = arr
        return input_data

    def predict_prob(self, frame_bgr):
        inp = self.preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        # handle uint8 output
        if out.dtype == np.uint8:
            prob = float(out[0][0]) / 255.0
        else:
            prob = float(out[0][0])
        # The model uses sigmoid output (0..1)
        return prob

# Utility to save snapshot & log
def save_and_log(frame, frame_id, label, prob, decision, enc_tick):
    ts = datetime.utcnow().isoformat()
    fname = f"{frame_id}_{int(time.time())}.jpg"
    path = os.path.join(SNAPSHOT_DIR, fname)
    cv2.imwrite(path, frame)
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{frame_id},{label},{prob:.3f},{decision},{enc_tick},{path}\n")
    return path

# Main loop
def main():
    global encoder_count

    # Setup GPIO, encoder, servo
    gpio_setup()
    start_encoder_listener()
    servo = Servo(SERVO_PIN)

    # Load model
    model = TFLiteModel(MODEL_PATH)

    # Camera setup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Camera not available.")
        return

    print("Starting main loop. Press Ctrl+C to stop.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, retrying...")
                time.sleep(0.2)
                continue

            frame_idx += 1
            # Optionally crop to region of interest where object passes
            # roi = frame[y1:y2, x1:x2]
            roi = frame

            prob = model.predict_prob(roi)  # 0..1 prob for class "biogas_suitable"
            label = "biogas_suitable" if prob >= 0.5 else "not_suitable"

            # Decision logic with thresholds
            if prob >= CONF_ACCEPT:
                decision = "auto_accept"
            elif prob < CONF_REJECT:
                decision = "auto_reject"
            else:
                decision = "audit"  # low confidence -> route to audit bin

            # Determine encoder tick now and compute target tick
            with encoder_lock:
                current_tick = encoder_count
            target_tick = current_tick + TICKS_TO_ACTUATOR

            # Save snapshot and log
            frame_id = f"frame{frame_idx:06d}"
            snap_path = save_and_log(roi, frame_id, label, prob, decision, current_tick)

            # publish via MQTT (optional)
            if MQTT_AVAILABLE:
                try:
                    payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "frame_id": frame_id,
                        "label": label,
                        "prob": float(prob),
                        "decision": decision,
                        "encoder_tick": int(current_tick),
                        "image_path": snap_path
                    }
                    publish.single("waste/sorter/classification", json.dumps(payload), hostname="localhost")
                except Exception as e:
                    print("MQTT publish failed:", e)

            # Wait until item reaches actuator
            # Simple busy wait; for efficiency use event-driven approach
            while True:
                with encoder_lock:
                    tick_now = encoder_count
                if tick_now >= target_tick:
                    break
                time.sleep(0.002)

            # Act according to decision
            # For auto_accept we do nothing (default conveyor path to accept)
            if decision == "auto_reject":
                # push to reject bin
                servo.push()
            elif decision == "audit":
                # push to audit bin (use same or alternate push depending on mech)
                servo.push()
            # else auto_accept: do nothing, item goes to accept bin

            # small delay to avoid rapid repeats
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        servo.cleanup()
        if GPIO:
            GPIO.cleanup()

if __name__ == "__main__":
    main()
