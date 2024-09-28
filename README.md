# Improved YOLOv8 with MS-CA-RFAConv Module and Custom Danger Zone Detection

SafeZone Surveillance is an advanced monitoring tool that integrates the YOLOv8 object detection model with customizable danger zones. It allows users to set up a real-time monitoring system using screen capture or a remote camera feed.
## How to start
pip install -r requirements.txt    (python version 3.11.x)
If lack please find requirementsbase.txt to ensure your missing pack version！！！QWQ（I always use my base env sry）
!pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html      
(IF YOU HAVE CUDA)
"!pip install requirementslinux-train.txt"
## Features

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

