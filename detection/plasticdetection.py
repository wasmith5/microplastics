# USAGE
# python detection/file_detection.py --input videos/people2.avi --output videos/output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import argparse
from cv2 import cv2
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'models/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('models', 'label_map.pbtxt')
PATH_TO_TEST_IMAGES_DIR = 'transferlearning/data/images/test'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=3, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required=True,
    help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=30,
    help="FPS of output video")
ap.add_argument("-co", "--codec", type=str, default="XVID",
    help="codec of output video")
ap.add_argument("-i", "--input", required=True,
    help="path to input video file")
args = vars(ap.parse_args())

# initialize the list of class labels that our model was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [ "fiber", "bead", "size" ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

ct = CentroidTracker()

print("[INFO] starting video...")
vs = cv2.VideoCapture(args["input"])
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None

total = 0

# loop over the frames from the video stream
while (vs.isOpened()):

    ret,frame = vs.read()

    if writer is None:
        w = vs.get(3)
        h = vs.get(4)
        w = int(w)
        h = int(h)
        writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w, h), True)

    if ret == True:

        frame_expanded = np.expand_dims(frame, axis=0)
        output_dict = run_inference_for_single_image(frame, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        boxes = output_dict['detection_boxes']
        draw = boxes.shape[0]
        scores = output_dict['detction_scores']
        min_score = args["confidence"]
        rects = []

        for i in range(min(draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score:
                class_name = category_index[output_dict['detection_classes'][i]]['name']

        writer.write(frame)

        cv2.imshow("Microplastics Detection", frame)
        key = cv2.waitkey(1) & 0xFF

        if key == ord("q"):
            break
    else:
        break

    

for key, value in areas.items():
    print("Area of ", key, ":", value)

print("Total detections: ", total+1)

# do a bit of cleanup
vs.release()
writer.release()
cv2.destroyAllWindows()