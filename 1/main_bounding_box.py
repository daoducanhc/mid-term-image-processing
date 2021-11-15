import cv2
import numpy as np

img = cv2.imread("1/img.jpg")
result_temp = img[830:, 1450:]
img = img[:, :1420]

result = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result_temp_gray = cv2.cvtColor(result_temp, cv2.COLOR_BGR2GRAY)

h, w = img.shape[:2]

for i in range(8,9):
    template = cv2.imread("1/dataset/{}.png".format(i), cv2.COLOR_BGR2GRAY)
    tH, tW = template.shape[:2]
    mask = cv2.imread("1/dataset/{}_mask.png".format(i))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    found = None

    # for scale in np.linspace(1.2, 2.2, 20):
    for scale in np.linspace(1.3, 1.7, 40):
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))

        corrimg = cv2.matchTemplate(resized, template, cv2.TM_CCORR_NORMED, mask=mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(corrimg)

        if found is None or max_val > found[0]:
            found = (max_val, max_loc, scale)

    (max_val, max_loc, scale) = found
    max_val_ncc = '{:.3f}'.format(max_val)
    print("correlation match score: " + max_val_ncc)

    (startX, startY) = (int(max_loc[0] / scale), int(max_loc[1] / scale))

    (endX, endY) = (int((max_loc[0] + tW) / scale), int((max_loc[1] + tH) / scale))

    cv2.rectangle(result, (startX, startY), (endX, endY), (0,255,0), 5)

    cv2.putText(result, str(i+1), (startX-30, startY-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

    corrimg = cv2.matchTemplate(result_temp_gray, template, cv2.TM_CCORR_NORMED, mask=mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(corrimg)

    (startX, startY) = (max_loc[0], max_loc[1])

    (endX, endY) = (max_loc[0] + tW, max_loc[1] + tH)

    cv2.rectangle(result_temp, (startX, startY), (endX, endY), (0,255,0), 5)

    cv2.putText(result_temp, str(i+1), (startX-20, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

result = cv2.resize(result, (600,600))
cv2.imshow('result', result)

result_temp = cv2.resize(result_temp, (600,600))
cv2.imshow('result_temp', result_temp)

cv2.waitKey(0)
# img = cv2.resize(img, (600,600))
# cv2.imshow("img", img)
# cv2.waitKey(0)