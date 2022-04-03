import cv2
filename = "latpull_back.mp4"
cap = cv2.VideoCapture(filename)

def rescale_frame(frame_input, percent=25):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

ret, frame = cap.read()
rescaled_frame = rescale_frame(frame)
(h, w) = rescaled_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("resize_"+filename,
                            fourcc, 30.0,
                            (w, h), True)
while cap.isOpened():
    ret, frame = cap.read()

    rescaled_frame = rescale_frame(frame)

    # write the output frame to file
    writer.write(rescaled_frame)

    cv2.imshow("Output", rescaled_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
writer.release()