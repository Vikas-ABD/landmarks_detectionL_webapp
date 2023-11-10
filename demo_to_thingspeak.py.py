import cv2
import torch
import numpy as np
import requests
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image

# Replace this with the path to your YOLO model
model = YOLO(f'best.pt')

# Define a function to process and modify the image from the camera
def process_frame():
    # Initialize the webcam
    videostream = cv2.VideoCapture(0)

    # Initialize variables for Opera_House detection count and ThinkSpeak update count
    opera_house_count = 0
    think_speak_count = 0

    while True:
        # Grab frame from the video stream
        ret, frame = videostream.read()

        # Make predictions with the YOLO model directly on the frame
        with torch.no_grad():
            results = model.predict(frame, conf=0.25)

        # Get the class predictions
        class_predictions = results[0].boxes.cls.cpu().numpy()

        # Loop through all detected classes
        for i, class_id in enumerate(class_predictions):
            label = model.names[int(class_id)]
            if label == 'Opera_House':
                opera_house_count += 1
                bbox = results[0].boxes.xyxy[i].cpu().numpy().astype(int)

                # Draw bounding boxes around the detected objects
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # If 20 consecutive frames contain Opera_House, send data to ThinkSpeak
        if opera_house_count >=2:
            think_speak_count += 1
            write_api_key = "IW5WG8QFV9FBMA1Y"
            url = "https://api.thingspeak.com/update"
            payload = {'api_key': write_api_key, 'field1': 1}  # Assuming the field is named 'field1'
            response = requests.get(url, params=payload)
            print(f"Data sent to ThinkSpeak server {think_speak_count} times.")

        # Display the frame
        cv2.imshow('Object Detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.release()

# Process the frame from the camera
process_frame()
