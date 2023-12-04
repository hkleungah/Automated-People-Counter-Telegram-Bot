# Automated-People-Counter-Telegram-Bot

## Overview
The People Counter Bot is an innovative tool designed for analyzing images and videos to count the number of people present. This project leverages advanced machine learning and computer vision techniques to provide accurate and timely assessments for various applications, such as crowd monitoring and traffic flow analysis.

## Features
- **Image Analysis**: Upload an image, and the bot will count the number of people in it, providing a quick crowd size assessment.
- **Video Analysis**: Upload a video, and the bot will count the number of people crossing a virtual line (set either vertically or horizontally) to understand movement patterns or traffic flow.

## Demo
Check out our [demo video](https://youtu.be/Ff2-4nvs7rU) to see the People Counter Bot in action.

## How It Works
1. **For Image Analysis**:
   - The bot receives an image from the user.
   - It processes the image to detect and count the number of people.
   - The result is sent back to the user.

2. **For Video Analysis**:
   - The user uploads a video and specifies the orientation of the counting line (vertical or horizontal).
   - The bot processes the video, setting up a virtual line.
   - It counts the number of people crossing this line during the video.
   - The final count, along with the processed video, is sent back to the user.

## Technologies Used
- Python
- OpenCV
- YOLO (You Only Look Once) for real-time object detection
- Telegram Bot API

## Setup and Installation
pip install -r requirements.txt

python app.py

or 

Build the Docker Image:

docker build -t people-counter-bot .

docker run -p 5000:5000 people-counter-bot



