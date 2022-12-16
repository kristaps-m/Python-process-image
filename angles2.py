import cv2
import numpy as np
import math

# read image
img = cv2.imread("TEST2.PNG")

# get color bounds of brown
lower =(0,30,60) # lower bound for each channel
upper = (20,50,80) # upper bound for each channel

# create the mask and use it to change the colors
thresh = cv2.inRange(img, lower, upper)

# apply morphology
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get contour
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
cntr = contours[0]

# get rotated rectangle from contour
rot_rect = cv2.minAreaRect(cntr)
box = cv2.boxPoints(rot_rect)
box = np.int0(box)

# draw rotated rectangle on copy of img
rot_bbox = img.copy()
cv2.drawContours(rot_bbox, [box], 0, (255,255,255), 1)

# get dimensions
(center), (width,height), angle = rot_rect

# print dimensions
print('length=', height)
print('thickness=', width)

# get center line from box
# note points are clockwise from bottom right
x1 = (box[0][0] + box[3][0]) // 2
y1 = (box[0][1] + box[3][1]) // 2
x2 = (box[1][0] + box[2][0]) // 2
y2 = (box[1][1] + box[2][1]) // 2

# draw centerline on image
center_line = img.copy()
cv2.line(center_line, (x1,y1), (x2,y2), (255,255,255), 1)

# compute center line length
cl_length = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
print('centerline_length',cl_length)

# write img with red rotated bounding box to disk
cv2.imwrite("blender_thresh.jpg", thresh)
cv2.imwrite("blender_morph.jpg", morph)
cv2.imwrite("blender_rot_rect.jpg", rot_bbox)
cv2.imwrite("blender_length.jpg", center_line)


# display it
cv2.imshow("THRESHOLD", thresh)
cv2.imshow("MORPH", morph)
cv2.imshow("BBOX", rot_bbox)
cv2.imshow("CENTLINE", center_line)
cv2.waitKey(0)