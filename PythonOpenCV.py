from __future__ import print_function
import numpy as np
import cv2
import math
"""For picture resizing"""
# import tkinter as tk
# root = tk.Tk()
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

def get_one_angle_using_all_three_sides(side1, side2, side3):
    angle = math.acos((side2**2 + side3**2 - side1**2) / (2.0 * side2 * side3))

    return math.degrees(angle)

def calc_distance_between_two_points(point_one, point_two):

    return ((point_two[0] - point_one[0]) ** 2 + (point_two[1] - point_one[1]) ** 2) ** 0.5

def distance_calculator(triangleVerticesList):
    # each point is a list with x and y coordinates e.g. [174, 233]
    point1 = triangleVerticesList[0][0]
    point2 = triangleVerticesList[1][0]
    point3 = triangleVerticesList[2][0]
    length1 = calc_distance_between_two_points(point1, point2)
    length2 = calc_distance_between_two_points(point2, point3)
    length3 = calc_distance_between_two_points(point3, point1)

    return [length1, length2, length3]

# load the image on disk.
image = cv2.imread('pictures_before_script/png_image.png')
#image = cv2.imread('pictures_before_script/test1.png')
#image = cv2.imread('pictures_before_script/test2.png')
#image = cv2.imread('pictures_before_script/test3.png')
#image = cv2.imread('pictures_before_script/test4.png')

cv2.imshow("Original", image)

# convert the color image into grayscale
grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find edges in the image using canny edge detection method
# Calculate lower threshold and upper threshold using sigma = 0.33
sigma = 0.33
v = np.median(grayScale)
low = int(max(0, (1.0 - sigma) * v))
high = int(min(255, (1.0 + sigma) * v))

edged = cv2.Canny(grayScale, low, high)

# After finding edges we have to find contours
# Contour is a curve of points with no gaps in the curve
# It will help us find location of shapes

# cv2.RETR_EXTERNAL is passed to find the outermost contours (because we want to outline the shapes)
# cv2.CHAIN_APPROX_SIMPLE is removing redundant points along a line
(cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
We are going to use contour approximation method to find vertices of
geometric shapes. The algorithm  is also known as Ramer Douglas Peucker alogrithm.
In OpenCV it is implemented in cv2.approxPolyDP method.abs

detectShape() function below takes a contour as parameter and
then returns its shape
 '''


def detectShape(cnt):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape is triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape and vertices
    return [shape, vertices]

# Now we will loop over every contour
# call detectShape() for it and
# write the name of shape in the center of image

# loop over the contours
# create new list where area is > 10
cnts2 = [c for c in cnts if cv2.contourArea(c) > 10]


for c in cnts2:
    # compute the moment of contour
    M = cv2.moments(c)
    # From moment, we can calculate area, centroid etc.
    # The center or centroid can be calculated as follows

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # call detectShape for contour c
        shape = detectShape(c)[0]
    else:
        # set values as what you need in the situation
        shape = "Unknown"
        cX, cY = 0, 0

    # Outline the contours
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # Write the name of shape in the center of shapes
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (175, 62, 223), 2)

    if(shape == "triangle"):
        the_vertices = detectShape(c)[1]
        D = distance_calculator(the_vertices)
        angleA = round(get_one_angle_using_all_three_sides(D[0], D[1], D[2]), 2)
        angleB = round(get_one_angle_using_all_three_sides(D[1], D[2], D[0]), 2)
        angleC = round(get_one_angle_using_all_three_sides(D[2], D[0], D[1]), 2)
        listOfAngles = [angleB, angleC, angleA]
        for i in reversed(range(3)):
            cv2.putText(image, f"{listOfAngles[i]}", (the_vertices[i][0][0], the_vertices[i][0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(image, f"Total objects: {len(cnts2)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)
    #imS = cv2.resize(image, (screen_width, screen_height))
    cv2.imshow("Image", image)

print()
print(f"Total number of object in 'png_image.png' is: {len(cnts2)}")

cv2.waitKey(0)
cv2.destroyAllWindows()