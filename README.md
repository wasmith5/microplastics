# microplastics
Repository for ECE 492/494 Team 2 Microplastics Detection Project

--------------------------------------------------------------------------------------------------------------------------------------------

Software Stack: (Windows 10 64-bit OS)

    Python 3.6.10 64-bit (Anaconda)
    TensorFlow 2.2.0-rc1
    OpenCV 4.2.0
    TensorFlow Object Detection API

--------------------------------------------------------------------------------------------------------------------------------------------

Plastic Detection:

    1. Run the plasticdetection.py file.
        a. Use --input /path/to/input/video to choose an input video or change the default within the code.
        b. Use --ouput /path/to/output/video to choose where to save the output video or change the default within the code.

    2. Statistics will print to the terminal including:
        a. Total detections.
        b. Total beads.
        c. Total fibers.
        d. Area and diameter of all detected beads.
        e. Length of all detected fibers.

    3. The output video can then be viewed within the directory specified by the code/line command.

--------------------------------------------------------------------------------------------------------------------------------------------

Training via Transfer Learning:

    1. Load training and testing images into transferlearning/data/raw.

    2. Run resize_images.py.
        a. Move resized images into either transferleanint/data/train or transferlearning/data/test.
    
    3. Run labelimg.exe software in order to label images for training/testing.
        a. A .xml file will be created for each image.

    4. Open the tensorflow_object_detection_training_colab.ipynb in Google Colab.
        a. Follow all of the steps in the notebook.
        b. You will end up downloading a frozen_inference_graph.pb, a label_map.pbtxt, and a ssd_mobilenet_v2_coco.config.
        c. Save these files in the models/ folder.

    5. Run tf_text_graph_ssd.py from within the opencv/sources/samples/dnn folder on your computer.
        a. Save the new label map in the models/ folder and name it new_label_map.pbtxt.
        b. This new label map can be used with OpenCV's readNetFromTensorflow() function.

--------------------------------------------------------------------------------------------------------------------------------------------


