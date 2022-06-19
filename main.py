import cv2
import numpy as np
import time

import argparse
import numpy as np
import time
from openal import *
import time
from openal import *
import cv2
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

def ObjDetect():
    # Load the YOLO model
    net = cv2.dnn.readNet("C:/Users/std848622/Desktop/Mega/PROJECT_2Dto3D/yolov3.weights", "C:/Users/std848622/Desktop/Mega/PROJECT_2Dto3D/yolov3.cfg")
    classes = []
    with open("C:/Users/std848622/Desktop/Mega/PROJECT_2Dto3D/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Load webcam
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    starting_time = time.time()
    frame_id = 0

    while True:
        # Read webcam
        _, frame = cap.read()
        frame_id += 1
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Visualising data
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (40, 670), font, .7, (0, 255, 255), 1)
        cv2.putText(frame, "press [esc] to exit", (40, 690), font, .45, (0, 255, 255), 1)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            print("[button pressed] ///// [esc].")
            print("[feedback] ///// Videocapturing succesfully stopped")
            break

    cap.release()
    cv2.destroyAllWindows()



def audio_3D():


    if __name__ == "__main__":
        x_pos = 5
        sleep_time = 5
        source = oalOpen("C:/Users/std848622/pythonProject2/Audio/CantinaBand60.wav")
        source.set_position([x_pos, 0, 0])
        source.set_looping(True)
        source.play()
        listener = Listener()
        listener.set_position([0, 0, 0])

        while True:
            source.set_position([x_pos, 0, 0])
            print("Playing at: {0}".format(source.position))
            time.sleep(sleep_time)
            x_pos *= -1

        oalQuit()



def divide_video_to_frames():
    vidcap = cv2.VideoCapture('C:/Users/std84266/PycharmProjects/pythonProject/Audio/airplane1.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def image_orientation():
    # Load the image
    img = cv.imread("C:/Users/std84266/PycharmProjects/pythonProject/frame6.jpg")

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    cv.imshow('Input Image', img)

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if width < height:
            angle = 90 - angle
        else:
            angle = -angle

        label = "  Rotation Angle: " + str(angle) + " degrees"
        textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
                               (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
        cv.putText(img, label, (center[0] - 50, center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv.imshow('Output Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the output image to the current directory
    cv.imwrite("min_area_rec_output.jpg", img)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #ObjDetect()
    audio_3D()
