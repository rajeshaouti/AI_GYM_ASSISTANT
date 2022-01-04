# importing modules
import cv2
import numpy as np
import cv2.aruco as aruco
import os



def calculate_angle(a):
    radians = np.arctan2(a[2][1] - a[1][1], a[2][0] - a[1][0]) - np.arctan2(a[0][1] - a[1][1], a[0][0] - a[1][0])
    angle = abs(np.abs(radians * 180.0 / np.pi) - 180)
    return round(angle)


def find_point_position(id,lmlist):
    point = (lmlist[id][1], lmlist[id][2])
    return point


def findpositions(id1, id2, id3,lmlist):
    point1 = (lmlist[id1][1], lmlist[id1][2])
    point2 = (lmlist[id2][1], lmlist[id2][2])
    point3 = (lmlist[id3][1], lmlist[id3][2])
    return point1, point2, point3


def findcentroid(id1, id2,lmlist):
    point1 = (lmlist[id1][1], lmlist[id1][2])
    point2 = (lmlist[id2][1], lmlist[id2][2])
    centroid_x = int((lmlist[id1][1] + lmlist[id2][1]) / 2)
    centroid_y = int((lmlist[id1][2] + lmlist[id2][2]) / 2)
    centroid = (centroid_x, centroid_y)
    return centroid


def plot_point(point,color,img):
    cv2.circle(img, point, 5, color, cv2.FILLED)
    return None


def plot_lines_3points(pt1, pt2, pt3,img):
    points = np.array([(pt1), pt2, (pt3)])
    cv2.drawContours(img, [points], 0, (255, 255, 255), 2)


def plot(point, color,angle,img):
    cv2.line(img, point[0], point[1], color, 2)
    cv2.line(img, point[1], point[2], color, 2)

    cv2.circle(img, point[0], 2, color, cv2.FILLED)
    cv2.circle(img, point[1], 2, color, cv2.FILLED)
    cv2.circle(img, point[2], 2, color, cv2.FILLED)

    # if angle:
    cv2.putText(img, str(angle), point[1],
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return None


def plot_bar(angle, angle_limits,img):
    per = np.interp(angle, angle_limits, (0, 100))
    bar = np.interp(angle, angle_limits, (400, 120))
    # counter logic
    if per == 100:
        barcolor = (0, 255, 0)
    else:
        barcolor = (0, 0, 255)

    # Setup status box
    cv2.rectangle(img, (25, 120), (55, 400), barcolor, 2)
    cv2.rectangle(img, (25, int(bar)), (55, 400), barcolor, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (25, 110), cv2.FONT_HERSHEY_PLAIN, 2, barcolor, 1, cv2.LINE_AA)
    return per, bar


def plot_bar_horizontal(distance,img,thigh_half_length,color_Head_thigh):
    dis_mod = abs(distance)

    if distance <= 0:
        dis_mod = abs(distance)
        per = np.interp(dis_mod, (0, 2 * thigh_half_length), (-100, 0))
        bar = np.interp(dis_mod, (0, 2 * thigh_half_length), (330, 260))

        cv2.rectangle(img, (260, 20), (400, 40), color_Head_thigh, 2)
        cv2.rectangle(img, (330, 20), (int(bar), 40), color_Head_thigh, cv2.FILLED)
    elif distance > 0:
        per = np.interp(dis_mod, (2 * thigh_half_length, 0), (0, 100))
        bar = np.interp(dis_mod, (0, 2 * thigh_half_length), (330, 400))

        cv2.rectangle(img, (260, 20), (400, 40), color_Head_thigh, 2)
        cv2.rectangle(img, (330, 20), (int(bar), 40), color_Head_thigh, cv2.FILLED)


def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    # print(bboxs)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def plot_ArucoMarkers(arucofound,img):

    for bbox, id in zip(arucofound[0], arucofound[1]):
        tl = bbox[0][0][0], bbox[0][0][1]
        tr = bbox[0][1][0], bbox[0][1][1]
        br = bbox[0][2][0], bbox[0][2][1]
        bl = bbox[0][3][0], bbox[0][3][1]

        marker_centroid = int((tl[0] + tr[0] + br[0] + bl[0]) / 4), int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
        cv2.circle(img, (marker_centroid[0], marker_centroid[1]), 3, (255, 0, 0), 3)
        lx = int(tl[0])
        ly = int(tl[1])
        rx = int(br[0])
        ry = int(br[1])

        if rx == lx or ry == ly:
            bounding_box = (lx, ly, 100, 100)

        else:
            bounding_box = (lx, ly, (rx - lx), (ry - ly))

    return bounding_box

def graph_plot():
    os.system("Squats_version_1_plotting.py")