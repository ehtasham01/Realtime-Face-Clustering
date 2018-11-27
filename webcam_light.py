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

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["./face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["./face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

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
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    encodingsSize = len(encodings)
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detectionspython encode_faces.ppython encode_faces.py --dataset dataset --encodings encodings.pickley --dataset dataset --encodings encodings.pickle
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # print(faceBlob)
            # print(vec)
            # if vec:
            # print(vec)
            encodings += list(vec)
            clt.fit(encodings)


            # perform classification to recognize the face
            name = "ID"

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {}".format(name, clt.labels_[encodingsSize+i])
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



    # compute the facial embedding for the face
    
    # for i, box in enumerate(boxes):
    #     text = str(clt.labels_[encodingsSize+i])

    #     (top, right, bottom, left) = box
    #     cv2.rectangle(frame, (right,top), (left, bottom), (0, 0, 255), 2)
    #     cv2.putText(frame, text, (right,top),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



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