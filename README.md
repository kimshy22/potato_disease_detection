## Potato Disease Detection Using Machine Learning and ROS2 Integration

This project focuses on building an intelligent system capable of detecting potato plant diseases — specifically early blight, late blight, and healthy plants — using machine learning and deploying the model through a ROS2 (Robot Operating System 2) framework.

### Journey and Development Process

The work began with training a machine learning model using image datasets of potato leaves. The goal was to enable the model to learn to differentiate between healthy leaves and those affected by early or late blight. After successful training and testing of the model, the next phase involved deployment.

During deployment, the system was designed to operate through a camera node capturing real-time frames and a detection node responsible for processing these images. Initially, the inference module was functional but would produce random predictions even when no plant was present in the frame. This issue was addressed by developing a green-percentage detection mechanism that ensures inference is only performed when the captured image contains sufficient vegetation.

If a certain percentage of green pixels is detected across several frames, the system triggers the inference service, which then classifies the plant as healthy, early blight, or late blight. If no significant green is found, the system responds with “No plant detected.”

This integration between the green-detection logic and the inference service greatly improves the reliability of the detection process and reduces false predictions. Continuous troubleshooting and testing — focusing on camera setup, lighting conditions, and ROS2 node communication — helped achieve a working system capable of stable and accurate performance in real-world scenarios.

### Features

Real-time camera feed analysis using ROS2.

Preprocessing with contrast enhancement and color normalization.

Green-percentage detection for filtering non-plant frames.

Machine-learning model inference for disease classification.

FastAPI backend for inference service handling.

### Technologies Used

ROS2 (Robot Operating System 2)

Python (OpenCV, NumPy, PyTorch, FastAPI)

Machine Learning (ResNet-18)

Computer Vision for preprocessing and green detection

### Future Improvements

Enhance lighting consistency to improve detection accuracy.

Extend dataset and retrain model for better generalization.

Add continuous inference monitoring and visual feedback.
