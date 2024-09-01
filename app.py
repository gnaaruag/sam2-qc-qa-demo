import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import replicate
import json
import numpy as np
import subprocess
# Execute init.sh
subprocess.run(["sh", "init.sh"])

# Rest of the code
def main():
	# Your existing code here
	# ...
	file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])
	
	checkbox1 = st.checkbox("Check for gloves")
	checkbox2 = st.checkbox("Check for overall cleanliness")
	
	if st.button("Submit"):
		process_data(checkbox1, checkbox2, file)
  
  
def get_init_frame(video_path, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(output_path), exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Initialize frame counter
    frame_count = 0
    
    # Read frames from the video
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Break the loop if no more frames are available
        if not ret:
            break
        
        # Check if the current frame is the 10th one
        if frame_count == 9:  # 9 because frame_count starts at 0
            # Save the 10th frame as a JPEG image
            output_file = os.path.join(output_path, f"{frame_count:05d}.jpg")
            cv2.imwrite(output_file, frame)
            break  # Exit the loop after saving the 10th frame
        
        # Increment the frame counter
        frame_count += 1
    
    # Release the video file
    video.release()

# Call the function in process_data

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
		"score_thr": 0.4,
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

def fetch_http_url_for_input():
    # til object store is figured out, this will be the input
    return "https://replicate.delivery/pbxt/LXHeP3A0Z2LV8KU0TLruS39fVY5ldqVnHABbv6yfUDXnmvKB/test_small.mp4"

def sam2_call(coords):
	coords=np.array(coords, dtype=int)
	input = {
		"mask_type": "highlighted",
		"video_fps": 30,
		"input_video": fetch_http_url_for_input(),
		"click_coordinates": str(coords.tolist())[1:-1],
		"click_labels": '1',
		"click_object_ids": "obj1",
		"output_frame_interval": 5,
	}
	print(input)
	output = replicate.run(
		"meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d",
		input=input
	)
	return output


def detect_object(frame):
	print('f')
	print(frame)
	input = {
		"input_media": frame,
		"class_names": "gloves",
		"return_json": True,
		"score_thr": 0.1,
	}
 
	output = replicate.run(
		"zsxkib/yolo-world:d232445620610b78671a7f288f37bf3baec831537503e9064afcf0bfd0f0a151",
		input=input
	)
	# Check if all objects contain "cls" : "gloves"
	print(output)
	for key, value in output.items():
		
			print('aa')
			print(value)
	return True


def parse_all_frames(frames):
	print(frames)
	object, all = 0, 0
	for frame in frames:
		all += 1
		if detect_object(frame):
			object += 1
	# Count the number of frames with objects
	object_count = sum(detect_object(frame) for frame in frames)
	
	# Calculate the percentage of frames with objects
	object_percentage = object_count / all
	print (object_percentage)
	# Check if the percentage is greater than or equal to 80%
	if object_percentage >= 0.8:
		return True
	else:
		return False
	

def process_data(checkbox1, checkbox2, file):
	if checkbox1:
		st.write("Checkbox 1 is selected")
	if checkbox2:
		st.write("Checkbox 2 is selected")
	if file is not None:
		file_path = os.path.join("uploads", file.name)
		with open(file_path, "wb") as f:
			f.write(file.read())
		st.write(f"File uploaded and stored at: {file_path}")
	
	# Call the function in process_data
	get_init_frame(file_path, os.path.join("uploads", "output_frames"))
	coords = generate_init(os.path.join("uploads", "output_frames", "00009.jpg"))
	plot = sam2_input(coords)
	show_plot(os.path.join("uploads", "output_frames", "00009.jpg"), plot)
	frames = sam2_call(plot)
	result = parse_all_frames(frames)
	if result:
		st.write("gloves detected")
	else:
		st.write("no gloves")
	
	
	
	

if __name__ == "__main__":
	main()