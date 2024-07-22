# SafeZone-Surveillance
SafeZone Surveillance is an advanced monitoring tool that integrates the YOLOv8 object detection model with customizable danger zones. It allows users to set up a real-time monitoring system using screen capture or a remote camera feed. The tool includes features such as alarm notifications and the ability to draw and manage custom danger zones for enhanced security.

Features
Model Selection: Choose your preferred YOLOv8 model for object detection.
Customizable Resolution: Set the resolution for the monitoring window to fit your needs.
Screen Capture Monitoring: Monitor your screen for any object detection.
Remote Camera Monitoring: Use a remote camera to capture and detect objects in real time.
Draw Danger Zones: Create custom danger zones using Bézier curves to specify areas of interest.
Alarm Notifications: Receive audio alerts when an object enters a predefined danger zone.
User-Friendly Interface: Simple and intuitive GUI for easy setup and management.
Installation
Clone the repository:
bash

git clone https://github.com/yourusername/safezone-surveillance.git
Navigate to the project directory:
bash

cd safezone-surveillance
Install the required dependencies:
bash

pip install -r requirements.txt
Run the application:
bash

python safezone_surveillance.py
Usage
Setup Configuration: Launch the application and configure your model path, resolution, and other settings.
Select Mode: Choose between screen capture or remote camera monitoring.
Draw Danger Zones: Enable draw mode and use your mouse to define danger zones.
Start Monitoring: Begin real-time object detection and receive alerts when objects enter danger zones.
Manage Danger Zones: Reset or modify danger zones as needed using the provided interface.
Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for more information.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
YOLOv8
OpenCV
Tkinter
