from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle
import csv
from datetime import datetime
import os

app = Flask(__name__)

# Load face encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Initialize webcam
video = cv2.VideoCapture(0)
attendance = set()  # To avoid duplicate entries

# Ensure attendance file exists
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

def mark_attendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])

def gen():
    while True:
        success, frame = video.read()
        print("Frame captured:", success)
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        names = []

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        for ((top, right, bottom, left), name) in zip(face_locations, names):
            # Draw rectangle and name
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown" and name not in attendance:
                attendance.add(name)
                mark_attendance(name)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

