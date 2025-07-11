import numpy as np
import tflite_runtime.interpreter as tf
import pygame
from pygame.locals import *
import cv2
import sys

# Initialize Pygame
pygame.init()
pygame.display.set_caption('Object Detection')

# Load the labels
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.Interpreter(model_path='detect.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video stream
cap = cv2.VideoCapture("rtsp://192.168.11.245:554/avstream/channel=1/stream=2.sdp")
ret, frame = cap.read()
if not ret:
    print("Error: Cannot open video stream.")
    sys.exit(1)

height, width = frame.shape[:2]
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
count = 0
running = True

while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        break

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE or event.key == K_q:
                running = False

    if count % 8 == 0:
        # Prepare frame for model
        input_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_data = np.expand_dims(input_frame.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        for i in range(len(scores)):
            if scores[i] > 0.21:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (
                    int(xmin * width),
                    int(xmax * width),
                    int(ymin * height),
                    int(ymax * height)
                )
                class_id = int(classes[i])
                label = labels[class_id] if class_id < len(labels) else 'Unknown'
                confidence = scores[i]

                # Draw bounding box and label BEFORE rotating
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert frame to RGB for Pygame display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.surfarray.make_surface(np.rot90(rgb_frame))  # Just rotate for display

        window.blit(pygame_frame, (0, 0))
        pygame.display.update()

    count += 1
    clock.tick(30)

cap.release()
pygame.quit()
cv2.destroyAllWindows()

