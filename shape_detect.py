import numpy as np
import matplotlib.pyplot as plt
import cv2

# read the image
image = cv2.imread('png_image.png')
cimg = image.copy()
# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(grayscale, 30, 100)

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)

# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=image, method=cv2.HOUGH_GRADIENT, dp=0.9,
                            minDist=80, param1=110, param2=39, maxRadius=70)

# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3)

for co, i in enumerate(circles[0, :], start=1):
    # draw the outer circle in green
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle in red
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

# print the number of circles detected
print("Number of circles detected:", co)

# show the image
plt.imshow(image)
plt.show()
