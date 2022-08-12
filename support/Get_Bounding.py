import numpy as np
import cv2

img = cv2.imread("B:\PycharmProjects\demo_OCR_Phenikka\Pyimagesearch\image_test\\2.jpg")
pts = np.array([[658, 1754], [1168, 1794], [1162, 1862], [652, 1822]])

## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x, y, w, h = rect
croped = img[y:y+h, x:x+w]

## (2) make mask
pts = pts - pts.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
bg = np.ones_like(croped, np.uint8)*255
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg+ dst


cv2.imwrite("croped.png", croped)
cv2.imwrite("mask.png", mask)
cv2.imwrite("dst.png", dst)
cv2.imwrite("dst2.png", dst2)