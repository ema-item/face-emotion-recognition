from insightface.app import FaceAnalysis
import streamlit as st
import subprocess
import json
import time
import cv2

################################################################ face detection model
# load model
model = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)

def face_detection(frame):
    if frame is None:
        return []
    # prediction
    faces = model.get(frame)
    boxes = []

    for face in faces:
        # bounding box extraction for a face
        x1, y1, x2, y2 = map(int, face.bbox)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes

####################################################################### emotion recognition

def run_em_detector(img_path):
    # emotion recognition model path
    em_script = "D:/privet/face_detection_model/emotion_recognition/em.py"
    # model with its own virtual environment and libraries
    em_python = "D:/privet/face_detection_model/emotion_recognition/env_emo/Scripts/python.exe"

    result = subprocess.run(
        [em_python, em_script, img_path],
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        raise RuntimeError(
            "External command returned no output.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )

    return json.loads(result.stdout.strip().splitlines()[-1])

################################################################## streamlit & infrence

st.title("Emotion Recognition Real-Time Webcam")
frame_placeholder = st.empty()


start = st.button("Start")
stop = st.button("Stop")


if "run" not in st.session_state:
    st.session_state.run = False

# start button
if start:
    st.session_state.run = True

# stop button
if stop:
    st.session_state.run = False

# start camera
if st.session_state.run:
    # reading frames from video or camera
    cam = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cam.read()
        if not ret:
            st.write("Camera error")
            break

        # save croped image path 
        img_path = "D:/privet/face_detection_model/frame_resource/croped_img.jpg"

        # face detection
        bboxes = face_detection(frame)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            # crop frame with bbox
            crop_frame = frame[y1:y2, x1:x2]

            # save croped image
            if frame is not None:
                cv2.imwrite(img_path, crop_frame)

            # face emorion recognition
            face_data = run_em_detector(img_path)
            
            # drawing face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # writing emotion
            cv2.putText(frame, face_data['label'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # writing confidence score
            cv2.putText(frame, str(face_data['score']), (x1, y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 255, 0), 2)

        # Update frame
        frame_placeholder.image(frame, channels="BGR")

        time.sleep(0.03)

        # stop
        if not st.session_state.run:
            break

    cam.release()
    cv2.destroyAllWindows()
