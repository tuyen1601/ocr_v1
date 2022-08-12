import cv2

def draw(img,line):

    img = cv2.line(img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), color=[255, 0, 0], thickness=2)
    img = cv2.line(img, (int(line[1][0]), int(line[1][1])), (int(line[2][0]), int(line[2][1])), color=[255, 0, 0], thickness=2)
    img = cv2.line(img, (int(line[2][0]), int(line[2][1])), (int(line[3][0]), int(line[3][1])), color=[255, 0, 0], thickness=2)
    img = cv2.line(img, (int(line[3][0]), int(line[3][1])), (int(line[0][0]), int(line[0][1])), color=[255, 0, 0], thickness=2)
    return img