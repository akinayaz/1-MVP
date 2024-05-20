import numpy as np
import cv2

temp = cv2.imread('test/test101.jpg')
#temp = cv2.imread("assets/inputs/images/robot.png") # assets/inputs/images/robotlar.png
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

ret, temp_thre = cv2.threshold(temp, 155, 255, 0)

cv2.imshow("store gray image", temp_thre)
cv2.waitKey(0)

#_, contours, hierarchy = cv2.findContours(temp_thre, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(temp_thre, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
temp_cont = sorted(contours, key=cv2.contourArea, reverse=True)

new_temp_cont = temp_cont[1]

tar = cv2.imread('test/test102.jpg')

#tar=cv2.imread("assets/inputs/images/robots.png")
tar_gray = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
tar_thr = cv2.threshold(tar_gray, 127, 255, 0)
#_, contours, hierarchy = cv2.findContours(tar_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(tar_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    match = cv2.matchShapes(new_temp_cont, c, 1, 0.0)
    print(match)
    if match < 0.15:
        closest_match = c
    else:
        closest_match = []
cv2.drawContours(tar, [closest_match], -1, (0, 255, 0), 2)
cv2.imshow("output", tar)
cv2.waitKey(0)
cv2.destroyAllWindows()