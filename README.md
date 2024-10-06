***HashTagify***



Flask-based web application that utilizes a combination of deep learning models for object detection, classification, and natural language processing (NLP) for image captioning. Here's a summary of the key components and functionalities:

1. **Flask Application**: Flask is a micro web framework for Python used to develop web applications. It provides tools, libraries, and technologies for building web applications.

2. **YOLOv5**: You're using YOLOv5, a state-of-the-art deep learning model for real-time object detection. YOLO (You Only Look Once) is known for its speed and accuracy in detecting objects within images.

3. **ResNet50**: ResNet50 is a convolutional neural network architecture known for its effectiveness in image classification tasks. You're using it to classify the objects detected by YOLOv5 into predefined categories.

4. **Hugging Face NLPConnect**: Hugging Face provides a powerful library for natural language processing (NLP) tasks. NLPConnect, a part of Hugging Face's offerings, likely integrates various NLP models and functionalities. You're utilizing it for image captioning, where it generates textual descriptions (captions) for the detected and classified objects in the images.

5. **Workflow**: The workflow of your application likely involves the following steps:
   - User uploads an image through the web interface.
   - YOLOv5 processes the image and detects objects within it.
   - Detected objects are then passed through ResNet50 for classification into specific categories.
   - Finally, NLPConnect generates captions for the detected and classified objects, providing textual descriptions of the contents of the image.

6. **User Interface**: The Flask application provides a user-friendly interface where users can upload images and receive captions for the objects detected within those images.

![multiple files result](https://github.com/himanshugupta11002/object_detection_classifcation_and_captioning/assets/72141497/e93fadfa-5a8c-4239-a495-521bc08e05be)
![home page](https://github.com/himanshugupta11002/object_detection_classifcation_and_captioning/assets/72141497/8c02f2d5-d856-45b7-ab18-3d5080cb9bd2)
![home page](https://github.com/himanshugupta11002/object_detection_classifcation_and_captioning/assets/72141497/e7521684-3c1a-40db-b0a3-6040248988bb)

<a href = 'https://drive.google.com/file/d/1T0YqbHcGGDni_OqWKeWUTpcwZf25Oik4/view?usp=sharing'>Drive link</a>

How to run application :
 1. set up project environment
 2. download yolov5 with its weight file yolov5s.pt
 3. download resnet50 with its weight file in .pth format
 4. install the necessary dependancies
 5. run the app.py file in the terminal with command flask app.py
