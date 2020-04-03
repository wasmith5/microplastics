# USAGE
# python detection/plasticdetection.py --input path/to/input/video --output path/to/output/video

# import the necessary packages
import numpy as np
import argparse
from cv2 import cv2
from pyimagesearch.centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.6,
    help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", default="videos/output.avi",
    help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=30,
    help="FPS of output video")
ap.add_argument("-co", "--codec", type=str, default="XVID",
    help="codec of output video")
ap.add_argument("-i", "--input", default="videos/captured001.mp4",
    help="path to input video file")
args = vars(ap.parse_args())

# initialize the list of class labels that our model was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [ " ", "bead", "fiber", "size" ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# declare the centroid tracker class used for tracking objects
ct = CentroidTracker()

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/new_label_map.pbtxt')

# initialize and preprocess the input video
print("[INFO] starting video...")
vs = cv2.VideoCapture(args["input"])
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None

# initialize the total detections counts
totalDetections = 0
detectedBeads = {}
detectedFibers = {}
fiberLengths = []
beadAreas = []

# loop over the frames from the video stream
while(vs.isOpened()):

    # grab the frame from the video and resize it
    # to have a maximum width of 400 pixels
    ret,frame = vs.read()

    if writer is None:
        w = vs.get(3)
        h = vs.get(4)
        w = int(w)
        h = int(h)
        writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h), True)
		
    if ret == True:

        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        rects = []
        indexes = []

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1]) 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                rects.append(box.astype("int"))
                indexes.append(idx)
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        objects = ct.update(rects, indexes)
        lengths = ct.getLengths()
        heights = ct.getHeights()
        indices = ct.getIndexes()

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if objectID > totalDetections:
                totalDetections = objectID
            if indices[objectID] == 1:
                detectedBeads[objectID] = 1
                beadAreas.append(lengths[objectID]*heights[objectID])
            if indices[objectID] == 2:
                detectedFibers[objectID] = 1
                fiberLengths.append(lengths[objectID])
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        writer.write(frame)

	    # show the output frame
        cv2.imshow("Plastic Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    else:
        break

totalBeads = len(detectedBeads)-1
totalFibers = len(detectedFibers)-1

print("[INFO] video ended...")

print("Total detections: ", totalDetections+1)
print("Total beads: ", totalBeads+1)
print("Total fibers: ", totalFibers+1)

for i in detectedFibers:
    print("Length of Fiber ", i, " = ", fiberLengths[i])
for i in detectedBeads:
    print("Area of Bead ", i, " = ", beadAreas[i][0])

# do a bit of cleanup
vs.release()
writer.release()
cv2.destroyAllWindows()