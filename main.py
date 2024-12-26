import cv2
import time
import os
import pygame
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8s_custom.pt')

# Initialize pygame mixer for sound
pygame.mixer.init()

# Open the video file
cap = cv2.VideoCapture('7317428-uhd_2160_3840_25fps.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

last_check_time = time.time()

# Create necessary directories
if not os.path.exists("WORKERS/NO_PERSON"):
    os.makedirs("WORKERS/NO_PERSON")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    results = model(frame,verbose=False)
    classes = []
    safety = ['Glass', 'Gloves', 'Helmet', 'Safety-Vest', 'helmet']
    person_detected = False  # Flag to check if a person is detected

    for r in results:
        for c in r.boxes:
            class_name = model.names[int(c.cls)]
            print(f"Detected: {class_name}")  # Debugging output to see what is being detected
            
            if class_name == 'person':  # Check if the detected class is "person"
                person_detected = True
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            elif class_name in safety:  # Detect safety items
                classes.append(class_name)
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Resize the frame if needed
    height, width = frame.shape[:2]
    frame_resized = cv2.resize(frame, (width // 3, height // 3))  # Resize to 33% of the original size

    if frame_resized is not None and frame_resized.size > 0:
        cv2.imshow('FRAME', frame_resized)
    else:
        print("Error: Empty frame.")
        break

    current_time = time.time()
    if current_time - last_check_time >= 15:
        if not person_detected:  # If no person is detected, play an alarm sound
            print("No person detected!")
            try:
                print("Attempting to play the alarm sound...")
                pygame.mixer.music.load('371836__iamgiorgio__j5_alarm-sound.mp3')  # Ensure you have the alarm sound file
                pygame.mixer.music.play()
                print("Alarm! No person detected.")
            except Exception as e:
                print(f"Error playing sound: {e}")
            
            # Save image if no person detected
            now = time.localtime()
            filename = f"WORKERS/NO_PERSON/{now.tm_year}{now.tm_mon}{now.tm_mday}_{now.tm_hour}{now.tm_min}{now.tm_sec}.jpg"
            cv2.imwrite(filename, frame)
        
        last_check_time = current_time

    # Check if 'q' key is pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
