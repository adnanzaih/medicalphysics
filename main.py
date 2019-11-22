'''
    This script uses the new Annual QC W-L phantom

    # Copyright (C) 2019 Adnan Hafeez
#
# # This program is a free software: you can redistribute it and/or modify
# # it under the terms of the GNU Affero General Public License as published
# # by the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version (the "AGPL-3.0+").
#
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU Affero General Public License and the additional terms for more
# # details.
#
# # You should have received a copy of the GNU Affero General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# # ADDITIONAL TERMS are also included as allowed by Section 7 of the GNU
# # Affero General Public License. These additional terms are Sections 1, 5,
# # 6, 7, 8, and 9 from the Apache License, Version 2.0 (the "Apache-2.0")
# # where all references to the definition "License" are instead defined to
# # mean the AGPL-3.0+.
#
# # You should have received a copy of the Apache-2.0 along with this
# # program. If not, see <http://www.apache.org/licenses/LICENSE-2.0>.
'''
import argparse
import pydicom
from skimage.morphology import square
from skimage.morphology import white_tophat
from skimage.filters.rank import bottomhat
import cv2
from tqdm import tqdm
import numpy as np
import time


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
    output = white_tophat(input, square(35))
    blur = cv2.GaussianBlur(output, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 254, 255, 0)
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
    cv2.putText(threshold, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Centroid", threshold)
    cv2.moveWindow("Centroid", 300, 30)
    print("X-Coordinate: "+str(cX))
    print("Y-Coordinate: "+str(cY))
    cv2.destroyWindow("Original Image")
    cv2.waitKey(0)
    return threshold, cX, cY
    #Done


def detectField(input):
    #radiation field detection and contouring
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = bottomhat(input, kernal)  # disk(30))
    blur = cv2.GaussianBlur(output, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 230, 255, 0)  # field
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
    cv2.putText(threshold, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Field", threshold)
    cv2.moveWindow("Field", 300, 30)
    print("X-Coordinate: " + str(cX))
    print("Y-Coordinate: " + str(cY))
    cv2.destroyWindow("Centroid")
    cv2.waitKey(0)
    return threshold, cX, cY
    #Done

def combinedPlots(out1,out2):
    cv2.imshow("Combined", out1[0]+out2[0])
    cv2.moveWindow("Combined", 300, 30)
    print("Centroid-Center X and Y Coordinates: ",out1[1],out1[2],"\n","Radiation Field Isocenter X and Y Coordinates: ", out2[1], out2[2])
    cv2.destroyWindow("Field")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-input', dest='input', help='sum the integers (default: find the max)', type=str)
results = parser.parse_args()
Analyze(results.input)


