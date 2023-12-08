import cv2

# Open the video file
video_path = 'calib_laptop_vid.webm'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a directory to save the frames (optional)
import os
output_directory = 'laptop_calibration_imgs'
os.makedirs(output_directory, exist_ok=True)

# Variables
frame_count = 0

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Save the frame
    frame_count += 1
    frame_filename = f"{output_directory}/frame_{frame_count:04d}.png"
    cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()

print(f"Frames extracted successfully. Total frames: {frame_count}")
