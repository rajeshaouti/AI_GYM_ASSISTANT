# Single Camera Capture

This program runs a single camera to analyze the side view of the human for Barbell sqaut performance tracking.

Instructions to run

```
pip install opencv-python
pip install opencv-contrib-python
pip install mediapipe
```

You can disable AruCo marker tracking at by setting

```
TRACK_ARUCO = False
```

After making sure that the camera index is matching with that of your camera. Run:

```
python Squats_Version_1.py
```

**Press Q to exit the program.**

This program runs a single camera to capture the side view human for Barbell sqaut performance tracking.
Aruco marker of dimension 6X6 is used for barbell tracking.The link for generating the aruco marker is,

```
https://chev.me/arucogen/
```
