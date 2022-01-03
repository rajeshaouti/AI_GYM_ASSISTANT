# importing modules
# !pip install opencv-python mediapipe

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
cap = cv2.VideoCapture(0)

# Run the code for plotting
_thread.start_new_thread(graph_plot, ())

# Creating a CSV file
num_coord = 33
landmarks = ["Point_no", "B_X0", "B_Y0"]
for val in range(1, num_coord + 1):
    landmarks += [f'x{val}', f'y{val}']

with  open('aruko_marker.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# Initial_Run for detecting the marker
initial_run_count = 10

while initial_run_count > 0:
    ok, img = cap.read()
    arucofound = findArucoMarkers(img)

    if len(arucofound[0]) != 0:
        bounding_box = plot_ArucoMarkers(arucofound, img)
        initial_run_count -= 1

    cv2.imshow("Tracking", img)
    cv2.waitKey(30)
cv2.destroyAllWindows()

#Detecting and tracking the marker
while cap.isOpened():
    try:
        ok = tracker.init(img, bounding_box)
    except:
        pass

    ok, img = cap.read()

    timer = cv2.getTickCount()

    arucofound = findArucoMarkers(img)

    if len(arucofound[0]) != 0:
        bounding_box = plot_ArucoMarkers(arucofound, img)
    else:
        try:
            ok, bounding_box = tracker.update(img)
        except:
            pass

    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # print(fps)

    # Draw bounding box
    if ok:

        if (int(bounding_box[0]) + int(bounding_box[2])) == int(bounding_box[0]) or (
                int(bounding_box[1]) + int(bounding_box[3])) == int(bounding_box[1]):
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[2]), int(bounding_box[3]))
        else:
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))

        centroid_tracking = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        cv2.circle(img, (centroid_tracking[0], centroid_tracking[1]), 3, (255, 0, 0), 3)

# Pose Dectection

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:

        lmlist = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])

        if len(lmlist) != 0:

            # Calculate angle back
            point_back = findpositions(11, 23, 25, lmlist)
            angle_back = calculate_angle(point_back)

            if angle_back < 125:  # EXPERT ADVICE
                color_back = color_green
            elif 135 > angle_back > 120:  # EXPERT ADVICE
                color_back = color_yellow
            else:
                color_back = color_red

            cv2.putText(img, str('Back'), (550, 40),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 25), (625, 50), color_back, cv2.FILLED)

            plot1 = plot(point_back, color_back, angle_back, img)

            # Calculate knee angle ,knee position ,toe position

            point_knee = findpositions(23, 25, 27, lmlist)
            angle_knee = calculate_angle(point_knee)
            knee_position = point_knee[1][0]

            point_toe = findpositions(25, 27, 31, lmlist)
            angle_toe = calculate_angle(point_toe)
            toe_position = point_knee[2][0]

            # Calculating knee overflow through foot distance
            ankle = find_point_position(27, lmlist)
            toe = find_point_position(31, lmlist)
            foot_length = int(math.sqrt((ankle[0] - toe[0]) ** 2 + (ankle[1] - toe[1]) ** 2))

            distance_knee_toe = abs(knee_position - toe_position)

            if distance_knee_toe < 1.1 * foot_length:  # EXPERT ADVICE
                color_knee = color_green
            elif distance_knee_toe < 1.3 * foot_length:  # EXPERT ADVICE
                color_knee = color_yellow
            else:
                color_knee = color_red

            cv2.putText(img, str('Knee'), (550, 90),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 75), (625, 100), color_knee, cv2.FILLED)
            plot2 = plot(point_knee, color_knee, abs(knee_position - toe_position), img)

            centroid_thigh = findcentroid(23, 25, lmlist)

            ear_position = find_point_position(7, lmlist)
            toe_position = find_point_position(31, lmlist)

            distance_H = (ear_position[0] - centroid_thigh[0])
            distance = abs(centroid_thigh[0] - ear_position[0])
            thigh_half_length = int(
                math.sqrt(
                    (point_knee[0][0] - centroid_thigh[0]) ** 2 + (point_knee[0][1] - centroid_thigh[1]) ** 2))

            if distance <= thigh_half_length:  # EXPERT ADVICE
                color_Head_thigh = color_green
                plot_point(centroid_thigh, color_Head_thigh, img)
                plot_point(ear_position, color_Head_thigh, img)
            elif distance <= 1.3 * thigh_half_length:  # EXPERT ADVICE
                color_Head_thigh = color_yellow
                plot_point(centroid_thigh, color_Head_thigh, img)
                plot_point(ear_position, color_Head_thigh, img)
            else:
                color_Head_thigh = color_red
                plot_point(centroid_thigh, color_Head_thigh, img)
                plot_point(ear_position, color_Head_thigh, img)

            cv2.putText(img, str('Head-Thigh'), (500, 140),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 125), (625, 150), color_Head_thigh, cv2.FILLED)

            # bounding box
            toe_1_position = find_point_position(29, lmlist)
            toe_2_position = find_point_position(31, lmlist)
            toe_3_position = find_point_position(30, lmlist)
            toe_4_position = find_point_position(32, lmlist)
            hip_position_1 = find_point_position(23, lmlist)
            hip_position_2 = find_point_position(24, lmlist)

            if toe_1_position > toe_2_position:  # left view
                rect_point_1 = int(toe_1_position[0] * 1.15), toe_1_position[1]
                rect_point_4 = int(toe_2_position[0] * 0.85), 10
                # distance_ear_and_bounding_box = (ear_position[0] - rect_point_1[0])
                # distance_hip_and_bounding_box = (hip_position_1[0] - rect_point_4[0])
                cv2.rectangle(img, rect_point_1, rect_point_4, color_Head_thigh, 1, cv2.LINE_AA)
            else:  # right view
                rect_point_1 = int(toe_3_position[0] * 0.85), toe_3_position[1]
                rect_point_4 = int(toe_4_position[0] * 1.15), 10
                # distance_ear_and_bounding_box = (ear_position[0] - rect_point_1[0])
                # distance_hip_and_bounding_box = (hip_position_2[0] - rect_point_4[0])
                cv2.rectangle(img, rect_point_1, rect_point_4, color_Head_thigh, 2, cv2.LINE_AA)

            plot_horizontal_column = plot_bar_horizontal(distance_H, img, thigh_half_length, color_Head_thigh)

            plot4 = plot_bar(angle_knee, (5, 110), img)  #  Expert Advice  - Angle limits
            color_list = [color_knee, color_Head_thigh, color_back]
            if plot4[0] == 100:
                if direction == 0:
                    direction = 1
                    count += 0.5
                    if color_red not in color_list:
                        good_count += 0.5
                    else:
                        good_count += 0
            if plot4[0] == 0:
                if direction == 1:
                    direction = 0
                    count += 0.5
                    if color_red not in color_list:
                        good_count += 0.5
                    else:
                        good_count += 0

            pose1 = results.pose_landmarks.landmark
            pose_data = list(
                np.array([[int((landmark.x) * w), int((landmark.y) * h)] for landmark in pose1]).flatten())
            dumbel_data = list(np.array([centroid_tracking[0], centroid_tracking[1]]))
            point_no = list(np.array([next(index)]))

            combined_data = point_no + dumbel_data + pose_data

            with  open('aruko_marker.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(combined_data)

            cv2.putText(img, 'Total_REPS', (25, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 5), (170, 35), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (130, 35),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, 'Good_REPS', (25, 75),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 50), (170, 80), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(good_count)), (130, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    #
    # ctime = time.time()
    # fps = 1 / (ctime - ptime)
    # ptime = ctime

    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
