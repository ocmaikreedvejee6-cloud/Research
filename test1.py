import cv2
import numpy as np
import time
import serial
import threading
import os
import smtplib
import requests
from datetime import datetime
from email.message import EmailMessage
from flask import Flask, Response
from pyngrok import ngrok

# ================= CONFIG =================
CONFIDENCE_THRESHOLD = 70
FACE_TIMEOUT = 3
PERSON_DETECT_INTERVAL = 3
VIDEO_TIMEOUT = 3

TELEGRAM_TOKEN = "8490765768:AAFU-Vpi0HAiS5_2V2mcboWYeiG8W4neiVE"
CHAT_ID = "7175315173"

EMAIL_ADDRESS = "growpfiveim312@gmail.com"
EMAIL_APP_PASSWORD = "qerlwnbhfcaprcll"
RECEIVER_EMAIL = "ocmaikreedvejee6@gmail.com"

NGROK_AUTH_TOKEN = "3CNooZSFRM64UqMFHQhvjL167bU_4RZuEZf7oztKsnwVyVcHJ"

# ================= UART =================
ESP32_PORT = "/dev/serial0"
ESP32_BAUD = 9600

esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
time.sleep(2)

def send_esp32(cmd):
    try:
        esp32.write((cmd + "\n").encode())
    except Exception as e:
        print("UART error:", e)

# ================= GLOBALS =================
last_face_time = 0
system_on = False

frame_global = None
lock = threading.Lock()

frame_count = 0

cap = None

# ================= NGROK =================
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ================= MODELS =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer1.yml")
label_map = np.load("labels1.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ================= CAMERA =================
def connect_camera():
    global cap
    while True:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            print("✅ Camera connected")
            return
        time.sleep(2)

# ================= FLASK STREAM =================
app = Flask(__name__)

def generate_frames():
    global frame_global
    while True:
        with lock:
            if frame_global is None:
                continue
            frame = frame_global.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ================= MAIN =================
def main():
    global last_face_time, system_on, frame_global, frame_count, cap

    print("🚀 System Running...")

    connect_camera()

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            time.sleep(2)
            connect_camera()
            continue

        frame = cv2.resize(frame, (640, 360))

        with lock:
            frame_global = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 6)
        face_detected = len(faces) > 0

        person_detected = False
        boxes = []

        frame_count += 1
        if frame_count % PERSON_DETECT_INTERVAL == 0:
            boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
            person_detected = len(boxes) > 0

        intruder = False

        # ================= FACE RECOGNITION =================
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                label, conf = recognizer.predict(face)
                name = label_map.get(label, "Unknown") if conf < CONFIDENCE_THRESHOLD else "Unknown"
            except:
                name = "Unknown"

            if name == "Unknown":
                intruder = True

        if person_detected and not face_detected:
            intruder = True

        now = time.time()

        # ================= RELAY CONTROL (UART ESP32) =================
        if person_detected or face_detected:
            last_face_time = now

            if not system_on:
                send_esp32("ON")
                system_on = True
                print("💡 RELAYS ON")

        else:
            if system_on and (now - last_face_time > FACE_TIMEOUT):
                send_esp32("OFF")
                system_on = False
                print("❌ RELAYS OFF")

        time.sleep(0.03)

# ================= START =================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    main()
