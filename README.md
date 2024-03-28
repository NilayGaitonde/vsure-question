# Yoga Pose Classifier with Feedback

This project is a deep learning-based application that can classify common yoga poses from video input and provide real-time feedback to users on how to improve their form. The application utilizes computer vision techniques to track body landmarks and a convolutional neural network (CNN) model to classify the poses.
The demo for this app can be found at : [demo](https://drive.google.com/file/d/1LuBbfA58cbf7IhakgDjrDaG002YIgBCx/view?usp=share_link)

## Features

- **Pose Classification**: The application can accurately classify three yoga poses: Tree Pose, Warrior Pose, and Downward-Facing Dog Pose, with an impressive 95% accuracy.
- **Real-time Feedback**: In addition to classifying the pose, the application provides real-time feedback to users on how to improve their form based on the detected body landmarks.
- **User-friendly Interface**: The application has a simple and intuitive user interface, making it easy for users to start the webcam and receive pose classification and feedback.

## Technologies Used

- **Deep Learning**: A CNN model was trained on a dataset of yoga pose images to achieve high classification accuracy.
- **Computer Vision**: The MediaPipe library was used for real-time body landmark detection and tracking.
- **Python**: The application was developed in Python, leveraging popular libraries such as OpenCV, TensorFlow, and NumPy.

## Installation

1. Clone the repository: `git clone https://github.com/nilaygaitonde/yoga-pose-classifier.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Run the application: `python capture.py`
2. The application will open a window displaying the webcam feed.
3. Perform one of the three supported yoga poses (Tree Pose, Warrior Pose, or Downward-Facing Dog Pose) in front of the webcam.
4. The application will classify the pose and provide real-time feedback on how to improve your form based on the detected body landmarks.

## Acknowledgments

- The yoga pose dataset used for training the CNN model was sourced from https://www.kaggle.com/datasets/nilaygaitonde/yoga-classification.
- The MediaPipe library for body landmark detection was developed by Google.
