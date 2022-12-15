from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

DISTANCE_CAMERA_WALL = 380 # mm

# Using Argument Parser to get the location of image
#img = cv2.imread('png_image.png')
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--png_image.png', required=True, help='Path to image') # 'png_image.png' # '--image'
# args = ap.parse_args()

# load the image on disk and then display it
image = cv2.imread('png_image.png') # 'png_image.png' # args.image raw_image.npy

cv2.imshow("Original", image)
plt.imshow(image)
plt.show()


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
# It will help us to find location of shapes

# cv2.RETR_EXTERNAL is passed to find the outermost contours (because we want to outline the shapes)
# cv2.CHAIN_APPROX_SIMPLE is removing redundant points along a line
#(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
We are going to use contour approximation method to find vertices of
geometric shapes. The alogrithm  is also known as Ramer Douglas Peucker alogrithm.
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

    # If the shape it triangle, it will have 3 vertices
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
    print("-----------------START-----------------")
    print(vertices, "vertaces", shape, "the shape", len(vertices), "Len-vertesies")
    print("-----------------END-----------------")

    # return the name of the shape
    return shape


# Now we will loop over every contour
# call detectShape() for it and
# write the name of shape in the center of image

C = 0
# loop over the contours
for c in cnts:
    # compute the moment of contour
    M = cv2.moments(c)
    # From moment we can calculte area, centroid etc
    # The center or centroid can be calculated as follows

    # cX = int(M['m10'] / M['m00'])
    # cY = int(M['m01'] / M['m00'])
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # call detectShape for contour c
        shape = detectShape(c)
    else:
        # set values as what you need in the situation
        shape = "Unknown"
        cX, cY = 0, 0

    # # call detectShape for contour c
    # shape = detectShape(c)

    # Outline the contours
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # Write the name of shape on the center of shapes
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)

print()
cv2.waitKey(0)
cv2.destroyAllWindows()