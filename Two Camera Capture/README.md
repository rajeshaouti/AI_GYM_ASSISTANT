# Two Camera Capture

This program runs a two camera to capture both the side view and front view of the human for Barbell sqaut performance tracking.
Aruco marker of dimension 6X6 is used for barbell tracking.The link for generating the aruco marker is,
```
https://chev.me/arucogen/
```
The program is set up in such a way that first it detects the marker and then starts huamn pose estimation.So a marker of the size 6X6 is required for performance tracking.
The marker dectection takes place from the side view cam.
```
Make sure that the camera index is matching with that of your camera in Squats_version_1_cam_0.py ,Squats_version_1_cam_0.py and
Squats_version_1_Two_camera_views.py. 

Run the Squats_version_1_Two_camera_views.py 

```
