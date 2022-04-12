import os
import threading


class CamThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    if camID == 1:
        os.system("Squats_Version_1_cam_1.py")
    else:
        os.system("Squats_version_1_cam_0.py")


# Create two threads as follows
thread1 = CamThread("Camera 1", 1)
thread2 = CamThread("Camera 2", 0)
thread1.start()
thread2.start()
