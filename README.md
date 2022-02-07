# AI_GYM_ASSISTANT

## AI Powdered Gym Assistant App
Introduction
There has been tremendous development in the field of online education which 
has enabled people to learn from the best sources, anywhere and at a less cost, 
but the same is not true for sports and physical training. The presence of a coach 
is always required which limits the reach and makes it more expensive.
This Project aim to build an AI-powered sports coaching app, which 
would give them real-time feedback on their performance and can predict any 
injuries that might occur due to erroneous practices.

## Problem Statement

We are focusing on developing an AI powdered Gym assistant app for the most 
important exercises such as squats, bench press, dumbbell clean and many more. 
There are a few human pose estimation models that are present in the market, but 
they are limited in only giving the location of the human body joints and don’t 
connect the pose estimation to correct postures for the required exercise. We want 
to address this issue and incorporate the domain knowledge from the sports 
industry and the technology available to make it more accessible to the public.

First, we are working on a Barbell Squat workout. The key parameters here are 
the back angle, knee angle, the position of head relative the thigh and the position
of barbell which is continuously monitored. The data from the domain expert is 
used to determine the threshold values of these parameters for a proper squat. The 
below image represents the screen shot of the program detecting a dumbbell 
squat. This model can track real-time human pose and indicate the correct 
postures for a dumbbell squat and tracks the repetitions.

The Visual indicators are used to give feedback to the user, and it is divided into 
green, yellow, and red if the posture is within the threshold limit, nearing the 
threshold limit and crossed the threshold limit respectively. 

The bounding box is a visual indicator for the person to be within the box.
A PVC pipe is used in place of a barbell for demonstration purpose and is traced 
using Aruco markers. 

The front view tracks the stance width relative to the shoulder length and the side view tracks all the parameters for a successful barbell 
squat. The graph represents the position of the barbell.

![Screenshot (13)](https://user-images.githubusercontent.com/26201695/152658957-dc9b420f-2136-4b32-bf2b-9534263c4f78.png)

Fig 1: Screen shot of the Real Time pose capture for a squat exercise.

The 3D reconstructed image is obtained by triangulation method using two 
cameras.

![Screenshot (14)](https://user-images.githubusercontent.com/26201695/152659129-f4f12ee5-ace9-4d9a-b8c8-aaf00bb93b80.png)

Fig 2: Screen shot of the Real Time 3D reconstruction of the human model performing a 
barbell squat.

# Requirement
```
Mediapipe
Python3.8
Opencv
matplotlib
```

## Future work 
• The feedback given to the user is now in the visual domain. We want to 
achieve the feedback as an audio and haptic feedback.

• To incorporate more exercises to the model and to create an app which can 
identify any given exercise and give real time feedback.

• To be able to predict any future injuries due to wrong posture
