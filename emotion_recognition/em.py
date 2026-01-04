from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import sys

def emotion_rec(frame):
    # load model
    model = load_model('D:/privet/face_detection_model/emotion_recognition/model/emotion_model.h5')
    # infrence
    preds = model.predict(frame)
    # emotions labels
    labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    # result label
    idx = int(np.argmax(preds[0]))
    # return result
    return {"label": labels[idx], "score": round(float(np.max(preds[0])), 2)}



if __name__ == "__main__":

    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.astype("float32") / 255.0
    img = img.reshape(1,48,48,1)
    # prediction
    result = emotion_rec(img)
    # return prediction result to face_detection file code with json file
    print(json.dumps(result))