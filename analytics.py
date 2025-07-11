import numpy as np
import tflite_runtime.interpreter as tf
import pygame
from pygame.locals import *
import cv2

# Initialize Pygame
pygame.init()
#window = pygame.display.set_mode((640, 480))

pygame.display.set_caption('Object Detection')

# Load the labels
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors.
interpreter = tf.Interpreter(model_path='detect.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video stream
# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("small.mp4") 
cap = cv2.VideoCapture("rtsp://192.168.11.245:554/avstream/channel=1/stream=2.sdp")  # Change to your video source
#ret, frame = cap.read()
#height, width, _ = frame.shape
#window = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
window = pygame.display.set_mode((600, 480))
count=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count%8==0:
    	# Preprocess the image
    	image = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    	input_data = image.astype(np.float32) / 255.0  # Convert to float32 and normalize
    	input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

	# Perform the inference
    	interpreter.set_tensor(input_details[0]['index'], input_data)
    	interpreter.invoke()

    	# Get the output
    	boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box 		coordinates of detected objects
    	classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of 		detected objects
    	scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of 		detected objects

    	# Process the results
    	for i in range(len(scores)):
        	if scores[i] > 0.21:  # Confidence threshold
            		ymin, xmin, ymax, xmax = boxes[i]
            		(left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1], 			ymin * frame.shape[0], ymax * frame.shape[0])
            		class_id = int(classes[i])
            		label = labels[class_id]
            		score = scores[i]

            		# Draw the bounding box and label
            		cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 		255, 0), 2)
            		cv2.putText(frame, f'{label}', (int(left), int(top) - 10), 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	# Display the output frame
    	#frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    	frame = cv2.flip(frame, 1)
    	frame = cv2.flip(frame, 0)
    	frame = cv2.flip(frame, 1)
    	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert image to RGB for Pygame
    	pygame_surface = pygame.surfarray.make_surface(frame)
    	# Rotate the Pygame surface by 90 degrees (Pygame's method of rotation)
    	pygame_surface_rotated = pygame.transform.rotate(pygame_surface, 90)
	
    	# Get the rectangle of the rotated surface to adjust the window size
    	rotated_rect = pygame_surface_rotated.get_rect(center=(window.get_width() // 2,      	window.get_height() // 2))
    	window.blit(pygame_surface, (0, 0))
    	#pygame.display.flip()
    	pygame.display.update()

    	for event in pygame.event.get():
        	if event.type == QUIT:
            		break
        	elif event.type == KEYDOWN:  # Key pressed
            		if event.key == K_ESCAPE:  # Check if the Escape key was pressed
                		break  
    count=count+1         		  

cap.release()
cv2.destroyAllWindows()
pygame.quit()
