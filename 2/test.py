import cv2
import numpy as np

img = cv2.imread("2/img.jpg")

template = img[250:, 1800:]
clone = template.copy()

template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = 255 - template

template = cv2.threshold(template, 20, 255, cv2.THRESH_BINARY)[1]

# template = cv2.Canny(template, 0, 100)
cv2.imshow("template", template)

_, contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

i=0
clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
cv2.imshow("clone", clone)
cv2.waitKey(0)

for c in contours:
    area = cv2.contourArea(c)
    if(area > 10000):
        (x, y, w, h) = cv2.boundingRect(c)
        height, width = clone.shape[:2]
        temp = np.zeros((height, width))
        cv2.drawContours(temp, [c], -1, (255, 255, 255), -1)
        temp = temp[y:y+h, x:x+w]
        cv2.imshow("temp", temp)

        temp1 = clone[y:y+h, x:x+w]
        cv2.imshow("temp1", temp1)
        cv2.waitKey(0)
        cv2.drawContours(clone, [c], -1, (0, 0, 0), -1)
        i+=1

print(i)

img = img[:, :1420]

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, 50, 200)
img = cv2.resize(img, (600,600))
cv2.imshow("img", img)

# template = cv2.Canny(template, 50, 200)
clone = 255 - clone

bb = clone.copy()
for c in contours:
    area = cv2.contourArea(c)
    if(area > 4000):
        (x, y, w, h) = cv2.boundingRect(c)
        temp1 = bb[y:y+h, x:x+w]

        temp = np.zeros((h, w))

        _, x, _ = cv2.findContours(temp1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("temp", temp)

        # cv2.waitKey(0)
        # cv2.rectangle(bb, (x, y), (x + w, y + h), (0, 255, 0), 2)

clone = cv2.resize(clone, (500,500))
cv2.imshow("clone", clone)
bb = cv2.resize(bb, (500,500))
cv2.imshow("bb", bb)

cv2.waitKey(0)
