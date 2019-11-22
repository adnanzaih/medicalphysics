'''
    This script uses the daily W-L phantom
    Hough Transform is used to detect circles
'''

import argparse
import pydicom
from skimage.morphology import square, disk
from skimage.morphology import black_tophat
from skimage.filters.rank import bottomhat
import cv2
from tqdm import tqdm
import numpy as np
import time
import math


def Analyze(input):
    filename = pydicom.read_file(input)
    img = filename.pixel_array
    cv2.imshow("Original Image", img)
    cv2.moveWindow("Original Image", 300, 30)
    cv2.waitKey(0)
#
    combinedPlots(
        detectCentroid(filename.pixel_array), detectField(filename.pixel_array)
    )



def detectCentroid(input):
    # Centroid Detection and contouring
    #Utilizing the Hough-Transform
    kernel = np.ones((4, 4), np.float32) / 25
    input = cv2.filter2D(input, -1, kernel)
    output = black_tophat(input, disk(15))
    blur = cv2.GaussianBlur(output, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 254, 255, 0)
    threshold = threshold.astype('uint8')
    centroid = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, 1, 100, param1=5, param2=10, minRadius=9, maxRadius=15)
    # ensure at least some circles were found
    if centroid is not None:
        # Convert the circle parameters a, b and r to integers.
        centroid = np.uint16(np.around(centroid))
        for pt in tqdm(centroid[0, :]):
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(threshold, (a, b), r, (78, 55, 128), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(threshold, (a, b), 1, (0, 0, 0), 3)
    print(centroid)
    cv2.destroyWindow("Original Image")
    cv2.imshow("Centroid", threshold)
    cv2.waitKey(0)
    return threshold, a, b


def detectField(input):
    #radiation field detection and contouring
    input = cv2.blur(input, (10, 10))
    kernel = np.ones((5, 5), np.float32) / 50
    input = cv2.filter2D(input, -1, kernel)
    output = bottomhat(input, square(3))  # disk(30))
    #blur = cv2.GaussianBlur(output, (5, 5), 0)
    _, threshold = cv2.threshold(output, 254, 255, 0)  # field
    threshold = threshold.astype('uint8')
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:

        for i in tqdm(np.arange(len(contours))):
            cv2.drawContours(threshold, [cnt], -1, (78, 55, -128), 1)
            M = cv2.moments(cnt)
            time.sleep(0.1)
        i = i + 1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(threshold, (cX, cY), 2, (78, 55, -128), 1)
    cv2.putText(threshold, "Radiation Field Iso-center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Field", threshold)
    cv2.moveWindow("Field", 300, 30)
    print("X-Coordinate: " + str(cX))
    print("Y-Coordinate: " + str(cY))
    cv2.destroyWindow("Centroid")
    cv2.waitKey(0)
    return threshold, cX, cY
    #Done


global window
def combinedPlots(out1,out2):
    distance = math.sqrt(((out1[1] - out2[1]) ** 2) + ((out1[2] - out2[2]) ** 2))
    cv2.imshow("Combined", out1[0]+out2[0])
    cv2.moveWindow("Combined", 300, 30)
    print("Centroid-Center X and Y Coordinates: ",out1[1],out1[2],"\n","Radiation Field Isocenter X and Y Coordinates: ", out2[1], out2[2])
    print("Distance between centers: ", distance)
    cv2.destroyWindow("Field")
    cv2.waitKey(0)
    cv2.destroyAllWindows()




parser = argparse.ArgumentParser()
parser.add_argument('-input', dest='input', help='sum the integers (default: find the max)', type=str)
results = parser.parse_args()
Analyze(results.input)


