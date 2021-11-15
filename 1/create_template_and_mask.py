import cv2
import numpy as np

img = cv2.imread("1/img.jpg")

template = img[830:, 1450:]

template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
clone = template.copy()

template = 255 - template

template = cv2.threshold(template, 20, 255, cv2.THRESH_BINARY)[1]

# cv2.imshow("template", template)
# cv2.waitKey(0)

_, contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

height, width = clone.shape[:2]

i=0
for c in contours:
    area = cv2.contourArea(c)
    if(area > 4000):
        (x, y, w, h) = cv2.boundingRect(c)

        mask = np.zeros((height, width))
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        mask = mask[y:y+h, x:x+w]
        cv2.imwrite('1/dataset/{}_mask.png'.format(i), mask)
        # cv2.imshow("mask", mask)

        temp = clone[y:y+h, x:x+w]
        cv2.imwrite('1/dataset/{}.png'.format(i), temp)
        # cv2.imshow("temp", temp)
        # cv2.waitKey(0)
        # cv2.drawContours(clone, [c], -1, (0, 0, 0), -1)
        # cv2.imshow("clone", clone)
        # cv2.waitKey(0)
        i+=1

print(i)
