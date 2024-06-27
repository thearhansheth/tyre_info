import pytesseract
import cv2 as cv
import sys
import os
from PIL import Image
import numpy as np

img = cv.imread("/Users/arhan.sheth/Documents/Codes/DX/tyre_info/z8YxC.jpg")
output_path = "/Users/arhan.sheth/Documents/Codes/DX/tyre_info/tyre_text"
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
center = (grayimg.shape[1] // 2, grayimg.shape[0] // 2)
max_radius = min(center[0], center[1])
circles = cv.HoughCircles(grayimg, cv.HOUGH_GRADIENT, 1, 20, param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv.circle(img, (i[0], i[1]), i[2],(0,255,0),2)

linear_image = cv.warpPolar(grayimg, (360, max_radius), center, max_radius, cv.WARP_POLAR_LINEAR)
cv.imshow("Original", img)
cv.imshow("Gray", grayimg)
cv.imshow("Linear", linear_image)
cv.waitKey(0)
cv.destroyAllWindows()
# output_file_path = os.path.join(output_path, "output.txt")
# with open(output_file_path, 'w') as f:
#     _, thresh = cv.threshold(grayimg, 127, 255, cv.THRESH_OTSU)
#     cv.imshow("Original", img)
#     cv.imshow("GrayImg", grayimg)
#     cv.imshow("Thresold image", thresh)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     text = pytesseract.image_to_string(thresh, config='-l eng')

#     f.write(text)
#     f.close()
#     print("Output Completed")

