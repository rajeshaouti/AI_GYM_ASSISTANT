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
import numpy as np
import json

import aigym

## AruCo Tracking
TRACK_ARUCO = False

#CONSTANTS
inf = float("inf")

# Initiation
index = count()
ptime = 0
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)
color_cyan = (255, 255, 0)
indicator_colors = [color_green,color_yellow,color_red,color_cyan]
good_count = 0
direction = 0
count = 0
point_no = []

EXERCISE = "lat_pull.json"

landmarkID = json.loads(open("mediapipe_landmarks.json").read())
exercise = json.loads(open(EXERCISE).read())
numberOfIndicators = len(exercise["indicators"])
indicator_status = [-1]*(numberOfIndicators+1)
total_sequences_length = len(exercise["sequence"])
present_sequence = 0
measurements = {"inf":inf,"-inf":-inf}
good_rep_count = 0
rep_count = 0


print("LOADING EXERCISE: "+exercise["name"])
print("VERSION: "+exercise["version"])

# mediapipe module
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Tracking the detected marker
tracker = cv2.TrackerCSRT_create()

# Capture the video feed
cap = cv2.VideoCapture("resize_latpull_back.mp4")

# Run the code for plotting aruco
_thread.start_new_thread(aigym.graph_plot, ())

# Creating a CSV file
num_coord = 33
landmarks = ["Point_no", "B_X0", "B_Y0"]
for val in range(1, num_coord + 1):
    landmarks += [f'x{val}', f'y{val}']

## ARUCO MARKER
if TRACK_ARUCO:

    with  open('aruko_marker.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)


    # Initial_Run for detecting the marker
    initial_run_count = 10

    while initial_run_count > 0:
        ok, img = cap.read()
        arucofound = aigym.findArucoMarkers(img)

        if len(arucofound[0]) != 0:
            bounding_box = aigym.plot_ArucoMarkers(arucofound, img)
            initial_run_count -= 1

        cv2.imshow("Tracking", img)
        cv2.waitKey(30)
    cv2.destroyAllWindows()


#Detecting and tracking the marker
while cap.isOpened():

    #ARUCO MARKER
    if TRACK_ARUCO:
        try:
            ok = tracker.init(img, bounding_box)
        except:
            pass
    
    #Updating the camera feed
    ok, img = cap.read()

    timer = cv2.getTickCount()

    #ARUCO MARKER
    if TRACK_ARUCO:
        arucofound = aigym.findArucoMarkers(img,draw=False)

        if len(arucofound[0]) != 0:
            bounding_box = aigym.plot_ArucoMarkers(arucofound, img)
        else:
            try:
                ok, bounding_box = tracker.update(img)
            except Exception as e: 
                print("Bounding box tracking",e)
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

    #detecting only if pose landmarks are present
    if results.pose_landmarks:
        h, w, c = img.shape
        landmark_list = []

        ind_text_start_x = 55
        ind_text_start_y = int(h*8/10)
        ind_box_start_x = 25
        ind_box_start_y = int(h*8/10)-15

        body_coordinates = {"x1":inf,"y1":-inf,"x2":-inf,"y2":inf}

        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, cx, cy])
            body_coordinates["x1"] = min(cx,body_coordinates["x1"])
            body_coordinates["y1"] = max(cy,body_coordinates["y1"])
            body_coordinates["x2"] = max(cx,body_coordinates["x2"])
            body_coordinates["y2"] = min(cy,body_coordinates["y2"])
        
        text_start_x = ind_text_start_x
        text_start_y = ind_text_start_y
        box_start_x = ind_box_start_x
        box_start_y = ind_box_start_y

        if len(landmark_list) != 0:
            #PREPROCESSING MEASUREMENTS
            for measurement in exercise["measurements"]:
                if measurement["type"] == "euclidean":
                    point1 = landmarkID[measurement["points"][0]]
                    point2 = landmarkID[measurement["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list)
                    if type(point2) == int:
                        point2 = aigym.find_point_position(point2,landmark_list)
                    measurements[measurement["name"]] = aigym.euclidean_distance(point1, point2)
                elif measurement["type"] == "absolute":
                    point1 = landmarkID[measurement["points"][0]]
                    point2 = landmarkID[measurement["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list)
                    if type(point2) == int:
                        point2 = aigym.find_point_position(point2,landmark_list)
                    measurements[measurement["name"]] = aigym.absolute_distance(point1, point2 ,axis = measurement["axis"])
                elif measurement["type"] == "angle":
                    point = aigym.findpositions(landmarkID[measurement["points"][0]], landmarkID[measurement["points"][1]], landmarkID[measurement["points"][2]], landmark_list)
                    measurements[measurement["name"]] = aigym.calculate_angle(point)
                elif measurement["type"] == "multiply":
                    measurements[measurement["name"]] = measurements[measurement["initial"]]*measurement["value"]
                elif measurement["type"] == "centroid":
                    landmarkID[measurement["name"]] = aigym.findcentroid(landmarkID[measurement["points"][0]], landmarkID[measurement["points"][1]], landmark_list)

            #UPDATING INDICATORS
            for indicator_index in range(numberOfIndicators):
                indicator = exercise["indicators"][indicator_index]
                if indicator["type"] == "angle":
                    point = aigym.findpositions(landmarkID[indicator["points"][0]], landmarkID[indicator["points"][1]], landmarkID[indicator["points"][2]], landmark_list)
                    angle = aigym.calculate_angle(point)
                    status = -1
                    box_color = color_red
                    if status == -1 and ("good" in indicator):
                        if angle <=indicator["good"].get("max",inf) and angle >= indicator["good"].get("min",-inf):
                            status = 0
                            box_color = color_green
                    
                    if status == -1 and ("intermediate" in indicator):
                        if angle <=indicator["intermediate"].get("max",inf) and angle >= indicator["intermediate"].get("min",-inf):
                            status = 1
                            box_color = color_yellow
                    
                    if status == -1:
                        status = 2
                        box_color = color_red
                    
                    plot1 = aigym.plot(point, box_color, angle, img)
                
                elif indicator["type"] == "relative":
                    attribute = measurements[indicator["name"]]
                    status = -1
                    box_color = color_red
                    if status == -1 and ("good" in indicator):
                        if attribute <=measurements[indicator["good"].get("max","inf")] and attribute >= measurements[indicator["good"].get("min","-inf")]:
                            status = 0
                            box_color = color_green
                    
                    if status == -1 and ("intermediate" in indicator):
                        if attribute <=measurements[indicator["intermediate"].get("max","inf")] and attribute >= measurements[indicator["intermediate"].get("min","-inf")]:
                            status = 1
                            box_color = color_yellow
                    
                    if status == -1:
                        status = 2
                        box_color = color_red
                    

                    
                indicator_status[indicator_index] = status
                
                cv2.putText(img, indicator["name"], (text_start_x, text_start_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
                text_start_y+=50
                cv2.rectangle(img, (box_start_x, box_start_y), (box_start_x+25, box_start_y+25), box_color, cv2.FILLED)
                box_start_y+=50
                
            
            ##COUNTING REPS
            sequence = exercise["sequence"][present_sequence]
            if sequence["type"] == "angle":
                point = aigym.findpositions(landmarkID[sequence["points"][0]], landmarkID[sequence["points"][1]], landmarkID[sequence["points"][2]], landmark_list)
                angle = aigym.calculate_angle(point)
                if angle <sequence.get("max",inf) and angle > sequence.get("min",-inf):
                    present_sequence = (present_sequence+1)%total_sequences_length
                    rep_count+=0.5
                    if 2 not in indicator_status:
                        good_rep_count+=0.5
            
            print("rep_count",rep_count,"good_rep_count",good_rep_count,indicator_status,angle,present_sequence)
            
            ##ADDITIONAL PLOTTING
            for plot in exercise["plot"]:
                if plot["type"] == "angle":
                    point = aigym.findpositions(landmarkID[plot["points"][0]], landmarkID[plot["points"][1]], landmarkID[plot["points"][2]], landmark_list)
                    angle = aigym.calculate_angle(point)
                    aigym.plot(point, indicator_colors[indicator_status[plot["indicator"]]], angle, img)
                
                elif plot["type"] == "point":
                    point = landmarkID[measurement["points"][1]]
                    if type(point) == int:
                        point = aigym.find_point_position(point1,landmark_list)
                    aigym.plot_point(point, indicator_colors[indicator_status[plot["indicator"]]], img)
                
                elif plot["type"] == "line":
                    point0 = landmarkID[plot["points"][0]]
                    if type(point0) == int:
                        point0 = aigym.find_point_position(point0,landmark_list)
                    point1 = landmarkID[plot["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list)
                    aigym.plot_lines_2_points(point0,point1, indicator_colors[indicator_status[plot["indicator"]]], img)



            # # Calculate angle back
            # point_back = aigym.findpositions(landmarkID["left_shoulder"], landmarkID["left_hip"], landmarkID["left_knee"], landmark_list)
            # angle_back = aigym.calculate_angle(point_back)

            # if angle_back < 125:  # EXPERT ADVICE
            #     color_back = color_green
            # elif 135 > angle_back > 120:  # EXPERT ADVICE
            #     color_back = color_yellow
            # else:
            #     color_back = color_red

            # cv2.putText(img, str('Back'), (550, 40),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.rectangle(img, (600, 25), (625, 50), color_back, cv2.FILLED)

            # plot1 = aigym.plot(point_back, color_back, angle_back, img)


            # # # Calculate knee angle ,knee position ,toe position
            # point_knee = aigym.findpositions(landmarkID["left_hip"], landmarkID["left_knee"], landmarkID["left_ankle"], landmark_list)
            # angle_knee = aigym.calculate_angle(point_knee)
            # knee_position = aigym.find_point_position(landmarkID["left_knee"], landmark_list)
            # knee_position_x = knee_position[0]
            # toe_position_x = aigym.find_point_position(landmarkID["left_foot_index"], landmark_list)[0]


            # # # Calculating knee overflow through foot distance
            # ankle = aigym.find_point_position(landmarkID["left_ankle"], landmark_list)
            # toe = aigym.find_point_position(landmarkID["left_foot_index"], landmark_list)
            # foot_length = int(math.sqrt((ankle[0] - toe[0]) ** 2 + (ankle[1] - toe[1]) ** 2))
            # foot_length1 = aigym.euclidean_distance(ankle, toe)

            # distance_knee_toe = abs(knee_position_x - toe_position_x)


            # # Updating KNEE indicators
            # # These can be updated to warn the user about posture as well
            # if distance_knee_toe < 1.1 * foot_length:  # EXPERT ADVICE
            #     color_knee = color_green
            # elif distance_knee_toe < 1.3 * foot_length:  # EXPERT ADVICE
            #     color_knee = color_yellow
            # else:
            #     color_knee = color_red

            # # #
            # # cv2.putText(img, str('Knee'), (550, 90),
            # #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            # # cv2.rectangle(img, (600, 75), (625, 100), color_knee, cv2.FILLED)
            # plot2 = aigym.plot(point_knee, color_knee, abs(knee_position_x - toe_position_x), img)

            # centroid_thigh = aigym.findcentroid(landmarkID["left_hip"], landmarkID["left_knee"], landmark_list)

            # ear_position = aigym.find_point_position(landmarkID["left_ear"], landmark_list)

            # distance_H = (ear_position[0] - centroid_thigh[0])
            # distance = abs(centroid_thigh[0] - ear_position[0])
            # thigh_half_length = int(
            #     math.sqrt(
            #         (point_knee[0][0] - centroid_thigh[0]) ** 2 + (point_knee[0][1] - centroid_thigh[1]) ** 2))
            
            # #Update HEAD-THIGH indicator
            # if distance <= thigh_half_length:  # EXPERT ADVICE
            #     color_Head_thigh = color_green
            #     aigym.plot_point(centroid_thigh, color_Head_thigh, img)
            #     aigym.plot_point(ear_position, color_Head_thigh, img)
            # elif distance <= 1.3 * thigh_half_length:  # EXPERT ADVICE
            #     color_Head_thigh = color_yellow
            #     aigym.plot_point(centroid_thigh, color_Head_thigh, img)
            #     aigym.plot_point(ear_position, color_Head_thigh, img)
            # else:
            #     color_Head_thigh = color_red
            #     aigym.plot_point(centroid_thigh, color_Head_thigh, img)
            #     aigym.plot_point(ear_position, color_Head_thigh, img)

            # cv2.putText(img, str('Head-Thigh'), (500, 140),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.rectangle(img, (600, 125), (625, 150), color_Head_thigh, cv2.FILLED)

            # Drawing a Bounding box full body

            # toe_1_position = aigym.find_point_position(landmarkID["left_heel"], landmark_list)
            # toe_2_position = aigym.find_point_position(landmarkID["left_foot_index"], landmark_list)
            # toe_3_position = aigym.find_point_position(landmarkID["right_heel"], landmark_list)
            # toe_4_position = aigym.find_point_position(landmarkID["right_foot_index"], landmark_list)

            # if toe_1_position > toe_2_position:  # left view
            #     rect_point_1 = int(toe_1_position[0] * 1.15), toe_1_position[1]
            #     rect_point_4 = int(toe_2_position[0] * 0.85), 10
            #     cv2.rectangle(img, rect_point_1, rect_point_4, color_green, 1, cv2.LINE_AA)
            # else:  # right view
            #     rect_point_1 = int(toe_3_position[0] * 0.85), toe_3_position[1]
            #     rect_point_4 = int(toe_4_position[0] * 1.15), 10
            #     cv2.rectangle(img, rect_point_1, rect_point_4, color_green, 2, cv2.LINE_AA)

            # plot_horizontal_column = aigym.plot_bar_horizontal(distance_H, img, thigh_half_length, color_green)


            ## Updating the FINAL count of the reps
            # plot4 = aigym.plot_bar(angle_knee, (5, 110), img)  #  Expert Advice  - Angle limits
            # color_list = [color_knee, color_Head_thigh, color_back]
            # if plot4[0] == 100:
            #     if direction == 0:
            #         direction = 1
            #         count += 0.5
            #         if color_red not in color_list:
            #             good_count += 0.5
            #         else:
            #             good_count += 0
            # if plot4[0] == 0:
            #     if direction == 1:
            #         direction = 0
            #         count += 0.5
            #         if color_red not in color_list:
            #             good_count += 0.5
            #         else:
            #             good_count += 0

            #DRAWING BODY BOUNDING BOX
            cv2.rectangle(img, (int(body_coordinates["x1"]*0.85),int(body_coordinates["y1"]*1.15)), 
            (int(body_coordinates["x2"]*1.15),int(body_coordinates["y2"]*0.85)), color_green, 1, cv2.LINE_AA)


            ## DATASET COLLECTION AND LOGGING
            pose1 = results.pose_landmarks.landmark
            pose_data = list(
                np.array([[int((landmark.x) * w), int((landmark.y) * h)] for landmark in pose1]).flatten())
            
            ## ARUCO MARKER
            if TRACK_ARUCO:
                dumbel_data = list(np.array([centroid_tracking[0], centroid_tracking[1]]))
                point_no = list(np.array([next(index)]))

                combined_data = point_no + dumbel_data + pose_data

                with  open('aruko_marker.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(combined_data)


            ## UPDATING REPS
            cv2.putText(img, 'TOTAL REPS', (25, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 5), (170, 35), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(rep_count)), (130, 35),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, 'GOOD REPS', (25, 75),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 50), (170, 80), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(good_rep_count)), (130, 80),
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
