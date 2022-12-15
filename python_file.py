import numpy as np
# pip install opencv-python==4.5.3.56
# pip install opencv-contrib-python = Successfully installed opencv-contrib-python-4.6.0.66
# pip install matplotlib
#import OpenCV
import cv2
from matplotlib import pyplot as pltd
#import big_fat_cock

print("TEST")

imaging = cv2.imread('png_image.png') # ,1

# Altering properties of image with cv2  
imaging_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)  
imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)  
# Importing Haar cascade classifier xml data  
xml_data = cv2.CascadeClassifier('XML-data.xml')  
# Detecting object in the image with Haar cascade classifier   
detecting = xml_data.detectMultiScale(imaging_gray, minSize = (30, 30))
# Amount of object detected  
amountDetecting = len(detecting)  
# Using if condition to highlight the object detected  
if amountDetecting != 0:  
    for (a, b, width, height) in detecting:  
        cv2.rectangle(imaging_rgb, (a, b), # Highlighting detected object with rectangle  
                      (a + height, b + width),   
                      (0, 275, 0), 9)  
# Plotting image with subplot() from plt  
pltd.subplot(1, 1, 1)  
# Displaying image in the output  
pltd.imshow(imaging_rgb)  
pltd.show()



"""
# Altering properties of image with cv2  
img_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)  
imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)  
# Plotting image with subplot() from plt  
pltd.subplot(1, 1, 1)  
# Displaying image in the output  
pltd.imshow(imaging_rgb)  
pltd.show()  


# cv2.imshow('image',img)  
# cv2.waitKey() # This is necessary to be required so that the image doesn't close immediately.  
# #It will run continuously until the key press.  
# cv2.destroyAllWindows()

# Set blobColor equal to zero to extract dark blobs
# and to extract light blobs,set it to 255

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
"""
