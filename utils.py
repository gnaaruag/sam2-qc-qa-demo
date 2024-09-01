import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import replicate
import json
import numpy as np

def chop_video_into_frames(video_path, output_path):
	# Create the output directory if it doesn't exist
	os.makedirs(os.path.join(output_path), exist_ok=True)
	
	# Open the video file
	video = cv2.VideoCapture(video_path)
	
	# Initialize frame counter
	frame_count = 0
	
	# Read frames from the video and save them as JPEG images
	while True:
		# Read the next frame
		ret, frame = video.read()
		
		# Break the loop if no more frames are available
		if not ret:
			break
		
		# Save the frame as a JPEG image
		output_file = os.path.join(output_path, f"{frame_count:05d}.jpg")
		cv2.imwrite(output_file, frame)
		
		# Increment the frame counter
		frame_count += 10
	
	# Release the video file
	video.release()

def show_plot(frame_path, coords):
	plt.figure(figsize=(12, 8))
	plt.title(f"frame {10}")
	plt.imshow(Image.open(frame_path))
	plt.scatter(coords[:, 0], coords[:, 1], marker='*', color='red')
	st.pyplot(plt)

def generate_init(init_frame):
	input_media = open(init_frame, 'rb')
	input = {
		"input_media": input_media,
		"class_names": "gloves",
		"return_json": True,
		"score_thr": 0.6,
		# "max_num_boxes": 100
		# "nms_thr": 0.5
	}
 
	output = replicate.run(
    "zsxkib/yolo-world:d232445620610b78671a7f288f37bf3baec831537503e9064afcf0bfd0f0a151",
    input=input
	)
	st.write(output)
	return output

def sam2_input(coords):
	data = json.loads(coords['json_str'])
	print(data)
	print(data.items())
	glove_instances = []
	for key, value in data.items():
		if value['cls'] == 'gloves':
			x0, y0, x1, y1 = value['x0'], value['y0'], value['x1'], value['y1']
			center_x = (x0 + x1) // 2
			center_y = (y0 + y1) // 2
			glove_instances.append([center_x, center_y])
	glove_instances = np.array(glove_instances)
	st.write(glove_instances)
	return glove_instances

def sam2_call():
    pass