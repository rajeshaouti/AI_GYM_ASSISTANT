import math
import cv2
import mediapipe as mp
import time
import _thread
import csv
import os
from itertools import count

from Squats_version_1_function_module import *

# Initiation
index = count()
ptime = 0
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)
good_count = 0
direction = 0
count = 0
point_no = []

# Openpose module
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Tracking the detected marker
tracker = cv2.TrackerCSRT_create()

# Capture the video feed
frame_shape = [640, 480]

cap = cv2.VideoCapture(1)
cap.set(3, frame_shape[1])
cap.set(4, frame_shape[0])

size = (frame_shape[0], frame_shape[1])

result = cv2.VideoWriter('frontview.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)


# Run the code for plotting
def graph_plot_cam_0():
    os.system("python3 Squats_version_2_FrontView.py")


# _thread.start_new_thread(graph_plot_cam_0, ())

# Creating a CSV file
num_coord = 33
landmarks = ["Point_no", "B_X0", "B_Y0"]
for val in range(1, num_coord + 1):
    landmarks += [f'x{val}', f'y{val}']

with  open('FrontView_data.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

# Initial_Run for detecting the marker
initial_run_count = 10

# Detecting and tracking the marker
while cap.isOpened():
    ok, img = cap.read()
    # Pose Detection
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        lmlist = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])

        if len(lmlist) != 0:
            # Shoulder Stance
            point_shoulder_1 = find_point_position(12, lmlist)
            point_shoulder_2 = find_point_position(11, lmlist)

            point_ankle_1 = find_point_position(28, lmlist)
            point_ankle_2 = find_point_position(27, lmlist)

            shoulder_width = abs(point_shoulder_1[0] - point_shoulder_2[0])
            ankle_width = abs(point_ankle_1[0] - point_ankle_2[0])

            stance_width = abs(shoulder_width - ankle_width)

            if stance_width < .1 * shoulder_width:
                color_shoulder = color_green
            elif stance_width < .3 * shoulder_width:
                color_shoulder = color_yellow
            else:
                color_shoulder = color_red

            plot_lines_2_points(point_shoulder_1, point_shoulder_2, color_shoulder, img)

    result.write(img)
    cv2.imshow('Tracking Marker FrontView', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
result.release()
cv2.destroyAllWindows()
