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
import threading
import pandas as pd

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

## TWO CAMERA TESTING

# EXERCISE_NAME = "leg_press_2.json"
# CAMERA_0 = "resize_legpress_top.mp4"
# CAMERA_1 = "resize_legpress_side.mp4"

EXERCISE_NAME = "lat_pull_2.json"
CAMERA_0 = "resize_latpull_back.mp4"
CAMERA_1 = "resize_latpull_side.mp4"

# EXERCISE_NAME = "squats_2.json"
# CAMERA_0 = "resize_edward1_squat_side.mp4"
# CAMERA_1 = "edward1_squat_front.mp4"

cameras = [CAMERA_0,CAMERA_1]


## SINGLE CAMERA TESTING

# EXERCISE_NAME = "leg_extension_1.json"
# CAMERA_0 = "resize_zuoan0_legextension_side.mov"

# cameras = [CAMERA_0]

EXERCISE = os.path.join("exercises",EXERCISE_NAME)


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

#GENERATING COLUMNS
columns = []
temp_columns = [(landmarkID[i],i) for i in landmarkID]
temp_columns.sort()
for col in temp_columns:
    columns.append(col[1]+"_1"+"_x")
    columns.append(col[1]+"_1"+"_y")
    columns.append(col[1]+"_1"+"_z")
    columns.append(col[1]+"_1"+"_v")
for col in temp_columns:
    columns.append(col[1]+"_2"+"_x")
    columns.append(col[1]+"_2"+"_y")
    columns.append(col[1]+"_2"+"_z")
    columns.append(col[1]+"_2"+"_v")

for indicator in exercise["indicators"]:
    columns.append(indicator["name"])

columns.append("sequence")

# mediapipe module
pose = []
# mpDraw1 = mp.solutions.drawing_utils
# mpPose1 = mp.solutions.pose
# pose1 = mpPose1.Pose()
# pose.append(pose1)

# mpDraw2 = mp.solutions.drawing_utils
# mpPose2 = mp.solutions.pose
# pose2 = mpPose2.Pose()
# pose.append(pose2)

class PoseLandmarks:
    pose = 0
    def __init__(self):
        self.self = self
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        self.pose = mpPose.Pose()
    def findLandmarks(self,results,cam,imgRGB):
        try:
            results[cam] = self.pose.process(imgRGB)
        except Exception as e:
            print("thread:",e)

def findLandmarks(pose,results,cam,imgRGB):
        try:
            results[cam] = pose.process(imgRGB)
        except e:
            print("thread:",e)

pose = [PoseLandmarks() for i in range(len(cameras))]

# Tracking the detected marker
tracker = cv2.TrackerCSRT_create()

# Capture the video feed
captures = []
for camera in cameras:
    captures.append(cv2.VideoCapture(camera))

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

data = []

def captureOpened(captures):
    opened = True
    for capture in captures:
        opened = opened & capture.isOpened()
    return opened

#Detecting and tracking the marker
while captureOpened(captures):

    #ARUCO MARKER
    if TRACK_ARUCO:
        try:
            ok = tracker.init(img, bounding_box)
        except:
            pass
    
    #Updating the camera feed
    ok = []
    img = []
    for capture in captures:
        k,im = capture.read()
        ok.append(k)
        img.append(im)

    if not all(ok):
        break

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
    imgRGB = []
    threads = []
    results = [0 for i in range(len(captures))]
    for i in range(len(img)):
        imgRGB.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
        t = threading.Thread(target=pose[i].findLandmarks, args=(results,i,imgRGB[i]))
        threads.append(t)

    for i in range(len(img)):
        threads[i].start()
    
    for i in range(len(img)):
        threads[i].join()

    #detecting only if pose landmarks are present
    if all([result_pose.pose_landmarks for result_pose in results]):
        landmark_list = [[] for i in range(len(imgRGB))]
        h = []
        w = []
        c = []
        body_coordinates = [
            {"x1":inf,"y1":-inf,"x2":-inf,"y2":inf},
            {"x1":inf,"y1":-inf,"x2":-inf,"y2":inf}
        ]

        for camera_index in range(len(imgRGB)):
            h.append(img[camera_index].shape[0])
            w.append(img[camera_index].shape[1])
            c.append(img[camera_index].shape[2])
            #landmarks for camera 0
            for id, lm in enumerate(results[camera_index].pose_landmarks.landmark):
                cx, cy = int(lm.x * w[camera_index]), int(lm.y * h[camera_index])
                landmark_list[camera_index].append([id, cx, cy])
                body_coordinates[camera_index]["x1"] = min(cx,body_coordinates[camera_index]["x1"])
                body_coordinates[camera_index]["y1"] = max(cy,body_coordinates[camera_index]["y1"])
                body_coordinates[camera_index]["x2"] = max(cx,body_coordinates[camera_index]["x2"])
                body_coordinates[camera_index]["y2"] = min(cy,body_coordinates[camera_index]["y2"])
        
        text_start_x = [55 for i in range(len(img))]
        text_start_y = [int(h[i]*8/10) for i in range(len(img))]
        box_start_x = [25 for i in range(len(img))]
        box_start_y = [int(h[i]*8/10)-15 for i in range(len(img))]

        if all([len(landmarkList) != 0 for landmarkList in landmark_list]):
            warning = "\n"
            #PREPROCESSING MEASUREMENTS
            for measurement in exercise["measurements"]:
                if measurement["type"] == "euclidean":
                    point1 = landmarkID[measurement["points"][0]]
                    point2 = landmarkID[measurement["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list[measurement["camera"]])
                    if type(point2) == int:
                        point2 = aigym.find_point_position(point2,landmark_list[measurement["camera"]])
                    measurements[measurement["name"]] = aigym.euclidean_distance(point1, point2)
                elif measurement["type"] == "absolute":
                    point1 = landmarkID[measurement["points"][0]]
                    point2 = landmarkID[measurement["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list[measurement["camera"]])
                    if type(point2) == int:
                        point2 = aigym.find_point_position(point2,landmark_list[measurement["camera"]])
                    measurements[measurement["name"]] = aigym.absolute_distance(point1, point2 ,axis = measurement["axis"])
                elif measurement["type"] == "diff":
                    diff = measurements[measurement["measurements"][0]] - measurements[measurement["measurements"][1]]
                    if measurement["abs"]:
                        diff = abs(diff)
                    measurements[measurement["name"]] = diff
                elif measurement["type"] == "angle":
                    point = aigym.findpositions(landmarkID[measurement["points"][0]], landmarkID[measurement["points"][1]], landmarkID[measurement["points"][2]], landmark_list[measurement["camera"]])
                    measurements[measurement["name"]] = aigym.calculate_angle(point)
                elif measurement["type"] == "multiply":
                    measurements[measurement["name"]] = measurements[measurement["initial"]]*measurement["value"]
                elif measurement["type"] == "centroid":
                    landmarkID[measurement["name"]] = aigym.findcentroid(landmarkID[measurement["points"][0]], landmarkID[measurement["points"][1]], landmark_list[measurement["camera"]])

            #UPDATING INDICATORS
            for indicator_index in range(numberOfIndicators):
                indicator = exercise["indicators"][indicator_index]
                if indicator["type"] == "angle":
                    point = aigym.findpositions(landmarkID[indicator["points"][0]], landmarkID[indicator["points"][1]], landmarkID[indicator["points"][2]], landmark_list[indicator["camera"]])
                    angle = aigym.calculate_angle(point)
                    status = -1
                    box_color = color_red
                    if status == -1 and ("good" in indicator):
                        if angle <=indicator["good"].get("max",inf) and angle >= indicator["good"].get("min",-inf):
                            status = 0
                            box_color = color_green
                            warning += indicator["good"].get("warning",indicator["name"]+" good")+"\n"
                    
                    if status == -1 and ("intermediate" in indicator):
                        if angle <=indicator["intermediate"].get("max",inf) and angle >= indicator["intermediate"].get("min",-inf):
                            status = 1
                            box_color = color_yellow
                            warning += indicator["intermediate"].get("warning",indicator["name"]+" average")+"\n"
                    
                    if status == -1:
                        status = 2
                        box_color = color_red
                        warning += indicator["bad"].get("warning",indicator["name"]+" bad")+"\n"
                    
                    plot1 = aigym.plot(point, box_color, angle, img[indicator["camera"]])
                
                elif indicator["type"] == "relative":
                    attribute = measurements[indicator["name"]]
                    status = -1
                    box_color = color_red
                    if status == -1 and ("good" in indicator):
                        if attribute <=measurements[indicator["good"].get("max","inf")] and attribute >= measurements[indicator["good"].get("min","-inf")]:
                            status = 0
                            box_color = color_green
                            warning += indicator["good"].get("warning",indicator["name"]+" good")+"\n"
                    
                    if status == -1 and ("intermediate" in indicator):
                        if attribute <=measurements[indicator["intermediate"].get("max","inf")] and attribute >= measurements[indicator["intermediate"].get("min","-inf")]:
                            status = 1
                            box_color = color_yellow
                            warning += indicator["intermediate"].get("warning",indicator["name"]+" average")+"\n"
                    
                    if status == -1:
                        status = 2
                        box_color = color_red
                        warning += indicator["bad"].get("warning",indicator["name"]+" bad")+"\n"
                    

                    
                indicator_status[indicator_index] = status
                
                cv2.putText(img[indicator["camera"]], indicator["name"], (text_start_x[indicator["camera"]], text_start_y[indicator["camera"]]), 
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
                text_start_y[indicator["camera"]]+=50
                cv2.rectangle(img[indicator["camera"]], (box_start_x[indicator["camera"]], box_start_y[indicator["camera"]]), 
                (box_start_x[indicator["camera"]]+25, box_start_y[indicator["camera"]]+25), box_color, cv2.FILLED)
                box_start_y[indicator["camera"]]+=50
                
            
            ##COUNTING REPS
            sequence = exercise["sequence"][present_sequence]
            if sequence["type"] == "angle":
                point = aigym.findpositions(landmarkID[sequence["points"][0]], landmarkID[sequence["points"][1]], landmarkID[sequence["points"][2]], landmark_list[sequence["camera"]])
                angle = aigym.calculate_angle(point)
                if angle <sequence.get("max",inf) and angle > sequence.get("min",-inf):
                    present_sequence = (present_sequence+1)%total_sequences_length
                    rep_count+=0.5
                    if 2 not in indicator_status:
                        good_rep_count+=0.5
            
            print("rep_count",rep_count,"good_rep_count",good_rep_count,indicator_status,present_sequence,warning)
            
            ##ADDITIONAL PLOTTING
            for plot in exercise["plot"]:
                if plot["type"] == "angle":
                    point = aigym.findpositions(landmarkID[plot["points"][0]], landmarkID[plot["points"][1]], landmarkID[plot["points"][2]], landmark_list[plot["camera"]])
                    angle = aigym.calculate_angle(point)
                    aigym.plot(point, indicator_colors[indicator_status[plot["indicator"]]], angle, img[plot["camera"]])
                
                elif plot["type"] == "point":
                    if "point" in plot:
                        point = landmarkID[plot["point"]]
                    else:
                        point = landmarkID[plot["measurement"]]
                    if type(point) == int:
                        point = aigym.find_point_position(point,landmark_list[plot["camera"]])
                    aigym.plot_point(point, indicator_colors[indicator_status[plot["indicator"]]], img[plot["camera"]])
                
                elif plot["type"] == "line":
                    point0 = landmarkID[plot["points"][0]]
                    if type(point0) == int:
                        point0 = aigym.find_point_position(point0,landmark_list[plot["camera"]])
                    point1 = landmarkID[plot["points"][1]]
                    if type(point1) == int:
                        point1 = aigym.find_point_position(point1,landmark_list[plot["camera"]])
                    aigym.plot_lines_2_points(point0,point1, indicator_colors[indicator_status[plot["indicator"]]], img[plot["camera"]])
            
            #DRAWING BODY BOUNDING BOX
            for camera_index in range(len(img)):
                cv2.rectangle(img[camera_index], (int(body_coordinates[camera_index]["x1"]*0.85),int(body_coordinates[camera_index]["y1"]*1.15)), 
                (int(body_coordinates[camera_index]["x2"]*1.15),int(body_coordinates[camera_index]["y2"]*0.85)), color_green, 1, cv2.LINE_AA)

            ## DATASET COLLECTION AND LOGGING FROM CAMERA
            pose_data = []
            for camera_id in range(len(img)):
                pose_results = results[camera_id].pose_landmarks.landmark
                # pose_data = list(np.array([[int((landmark.x) * w), int((landmark.y) * h)] for landmark in pose1]).flatten()
                for landmark in pose_results:
                    pose_data.extend([landmark.x,landmark.y,landmark.z,landmark.visibility])
                
            pose_data.extend(indicator_status[:-1])
            pose_data.extend([present_sequence])
            data.append(np.array(pose_data))

            ## ARUCO MARKER
            if TRACK_ARUCO:
                dumbel_data = list(np.array([centroid_tracking[0], centroid_tracking[1]]))
                point_no = list(np.array([next(index)]))

                combined_data = point_no + dumbel_data + pose_data

                with  open('aruko_marker.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(combined_data)


            ## UPDATING REPS
            cv2.putText(img[0], 'TOTAL REPS', (25, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img[0], (120, 5), (170, 35), (0, 0, 0), cv2.FILLED)
            cv2.putText(img[0], str(int(rep_count)), (130, 35),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img[0], 'GOOD REPS', (25, 75),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img[0], (120, 50), (170, 80), (0, 0, 0), cv2.FILLED)
            cv2.putText(img[0], str(int(good_rep_count)), (130, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

    for camera_index in range(len(img)):
        cv2.imshow('image'+str(camera_index), img[camera_index])

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
for capture in captures:
    capture.release()
cv2.destroyAllWindows()
data = np.array(data)
data = pd.DataFrame(data,columns = columns)
data.to_csv("_".join(CAMERA_0.split("_")[:-1])+"_data.csv",index=False)