import time
from math import *


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    a = a*180/pi
    if a >= 180:
        a = a-360
    return a

def getAngleBetweenPoint(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(atan2(deltaY, deltaX))
def getAngleBetweenPoints(line):

    deltaY = int(line[1][1]) - int(line[0][1])
    deltaX = int(line[1][0]) - int(line[0][0])
    angle1 = angle_trunc(atan2(deltaY, deltaX))

    deltaY = int(line[2][1]) - int(line[3][1])
    deltaX = int(line[2][0]) - int(line[3][0])
    angle2 = angle_trunc(atan2(deltaY, deltaX))

    return (angle1+angle2)/2

def getPointRotate(point, b, center):
    y = point[0]*cos(b) +  point[1]*sin(b) + center[1]
    x = point[1]*cos(b) - point[0]*sin(b) + center[0] 
    return [int(y),int(x)]

def getPoint_Center(pointA, pointO):
    y = pointA[0] -pointO[1]  
    x = pointA[1] -pointO[0]
    point = [y,x]
    return point

def getIntersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('Lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [int(x), int(y)]

# print(getIntersection([[5, 5], [-5, -5]], [[5, 0], [4, 0]]))
# angle = getAngleBetweenPoint(676, 234, 2095, 217)
# print(angle)

# angle1 = getAngleBetweenPoints([[675.0, 149.0], [2094.0, 132.0], [2095.0, 217.0], [676.0, 234.0]])
# angle2 = getAngleBetweenPoints([[951.0, 230.0], [1813.0, 217.0], [1814.0, 289.0], [952.0, 302.0]])
# print(angle1)