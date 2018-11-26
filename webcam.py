from imutils.video import FileVideoStream
from imutils.video import VideoStream

from sklearn.cluster import DBSCAN
from imutils.video import FPS
from imutils import paths
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os



print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

print("[INFO] clustering...")
clt = DBSCAN(min_samples=8, metric='euclidean', eps= 0.45, n_jobs=-1)

# start the FPS throughput estimator
fps = FPS().start()

encodings = []

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,model="cnn")

    encodingsSize = len(encodings)

    # compute the facial embedding for the face
    encoding = face_recognition.face_encodings(rgb, boxes)
    if encoding:
        encodings += encoding
        clt.fit(encodings)

    for i, box in enumerate(boxes):
        text = str(clt.labels_[encodingsSize+i])

        (top, right, bottom, left) = box
        cv2.rectangle(frame, (right,top), (left, bottom), (0, 0, 255), 2)
        cv2.putText(frame, text, (right,top),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()