# Improved YOLOv8 with MS-CA-RFAConv Module and Custom Danger Zone Detection

Improved YOLOv8 with MS- CA-RFAConv
Module and Custom Danger Zone Detection is an advanced monitoring tool that integrates the YOLOv8 object detection model with customizable danger zones. It allows users to set up a real-time monitoring system using screen capture or a remote camera feed.
## How to start
pip install -r requirements.txt    (python version 3.11.4)
If lack please find requirementsbase.txt to ensure your missing pack version！！！QWQ（I always use my base env sry）
!pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html      
(IF YOU HAVE CUDA)
"!pip install requirementslinux-train.txt"

## Implements
Run app.py and load your moudule

## Features
![检测对比](https://raw.githubusercontent.com/Qzm-QAQ/YOLOv8-with-MS-CA-RFAConv-Module-Custom-Danger-Zone-Detection-/refs/heads/main/assets/comparsion.jpg)
The right side represents the detection performance after adding MS-CA-RFAConv to the YOLO model.It's better to foucs on partial feature in detecting.
Use this System you can also customize line detection easily by user.
- Model Selection: Choose your preferred YOLOv8 model for object detection.
- Customizable Resolution: Set the resolution for the monitoring window.
- Screen Capture Monitoring: Monitor your screen for object detection.
- Remote Camera Monitoring: Use a remote camera to capture and detect objects in real-time.
- Draw Danger Zones: Create custom danger zones using Bézier curves.
- Alarm Notifications: Receive audio alerts when an object enters a danger zone.
- User-Friendly Interface: Simple and intuitive GUI for easy setup and management.
## Train with Rfaconv module
Please put your dataset in 'train'(formmat YOLO) dic and run the 'train.py'
## author：QIN ZIMING

