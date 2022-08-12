import numpy as np
import cv2
from support.Rotate import rotate

def Crop_Image(img, line,angle):

    remain_x = 5
    remain_y = 3
    h, w = img.shape[0], img.shape[1]
    # x_min = min(int(line[0][0]), int(line[3][0]))
    # y_min = min(int(line[0][1]), int(line[1][1]))
    # x_max = max(int(line[1][0]), int(line[2][0]))
    # y_max = max(int(line[2][1]), int(line[3][1]))
    # crop = img[y_min:y_max, x_min-10:x_max+10].copy()
    if int(line[0][0]) -remain_x < 0: x0 = 0
    else: x0 = int(line[0][0]) -remain_x
    if int(line[0][1]) -remain_y < 0: y0 = 0
    else: y0 = int(line[0][1]) -remain_y
    if int(line[1][0]) +remain_x > w: x1 = w
    else: x1 = int(line[1][0]) +remain_x
    if int(line[1][1]) -remain_y < 0: y1 = 0
    else: y1 = int(line[1][1]) -remain_y
    if int(line[2][0]) +remain_x > w: x2 = w
    else: x2 = int(line[2][0]) +remain_x
    if int(line[2][1]) +remain_y > h: y2 = h

    else: y2 = int(line[2][1]) +remain_y
    if int(line[3][0]) -remain_x < 0: x3 = 0
    else: x3 = int(line[3][0]) -remain_x
    if int(line[3][1]) +remain_y > h : y3 = h
    else: y3 = int(line[3][1]) +remain_y

    # pts = np.array([[int(line[0][0]), int(line[0][1])], [int(line[1][0]), int(line[1][1])],
    #                 [int(line[2][0]), int(line[2][1])], [int(line[3][0]), int(line[3][1])]])
    pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    crop = img[y:y + h, x:x + w]
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    pts = pts - pts.min(axis=0)
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(crop, crop, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(crop, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    rotated = rotate(dst2, angle, (0, 0, 0))
    bg = rotate(bg, angle, (0, 0, 0))

    ret, binary = cv2.threshold(bg, 254, 255, cv2.THRESH_BINARY_INV)

    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    x, y, width, height = cv2.boundingRect(cnts[0])

    result = rotated[y:y+height, x:x+width]
    return result, height, width