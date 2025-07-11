import face_recognition
import os
import pickle

known_encodings = []
known_names = []

path = "known_faces"
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(f"{path}/{filename}")
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"[WARNING] No faces found in {filename}")

data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encodings saved to encodings.pickle")
