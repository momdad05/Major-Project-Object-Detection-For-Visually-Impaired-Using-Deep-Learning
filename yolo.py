# USAGE
# python yolo_video_live.py --yolo yolo-coco

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pyttsx3
import threading

# Global variables for voice control
current_announcement = None
announcement_lock = threading.Lock()
engine = None

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolo.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
vs = cv2.VideoCapture(0)
(W, H) = (None, None)

# Initialize text-to-speech engine
def init_engine():
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)

init_engine()

def make_announcement(label, confidence):
    global current_announcement
    with announcement_lock:
        if current_announcement is None:
            current_announcement = (label, confidence)
            announcement = f"{label} detected with {confidence*100:.1f}% confidence"
            engine.say(announcement)
            engine.runAndWait()
            current_announcement = None

# Main processing loop
try:
    while True:
        # Read frame
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # Initialize frame dimensions if needed
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Process frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Detection variables
        boxes = []
        confidences = []
        classIDs = []
        best_detection = None
        max_confidence = args["confidence"]

        # Process detections
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # Track best detection
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_detection = (LABELS[classID], confidence, (x, y, width, height))

        # Apply NMS and draw boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Make announcement if new best detection
        if best_detection and (current_announcement is None or 
                             best_detection[0] != current_announcement[0]):
            label, confidence, _ = best_detection
            threading.Thread(target=make_announcement, args=(label, confidence), daemon=True).start()

        # Highlight best detection
        if best_detection:
            _, _, (x, y, w, h) = best_detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Display frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Program interrupted by user")

# Cleanup
vs.release()
cv2.destroyAllWindows()
engine.stop()
