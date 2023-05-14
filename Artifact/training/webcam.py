import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
physical_devices = tf.config.list_physical_devices("GPU")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


model = keras.models.load_model("Artifact/training/my_model")

labels = ['fist','okay','peace','point','point sideways','rock','stop','thumbs up']

cap = cv2.VideoCapture(0)
while cap.isOpened():

    _ , frame = cap.read()
    frame = frame [50:500, 50:500,:] 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resize = cv2.resize(rgb, (254,254))
    final = np.expand_dims(resize/255,0)

    value = model.predict(final)
    label = labels[np.argmax(value)]

    print(value)

    cv2.putText(frame, label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA )

    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        
        break

cap.release()
cv2.destroyAllWindows()
