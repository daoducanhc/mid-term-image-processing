import cv2
import numpy as np

img = cv2.imread("2/img.jpg")

template = img[550:840, 2240:2470]

template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
clone = template.copy()
template = 255 - template

template = cv2.threshold(template, 20, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("template", template)
cv2.waitKey(0)

_, contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

height, width = clone.shape[:2]

mask = np.zeros((height, width))
cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

cv2.imshow("mask", mask)
cv2.imshow("clone", clone)
cv2.waitKey(0)

cv2.imwrite('2/dataset/11_mask.png', mask)
cv2.imwrite('2/dataset/11.png', clone)
