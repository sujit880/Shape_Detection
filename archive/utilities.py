import cv2

import numpy as np


class point():
    x: int
    y: int 

def crop(image, mask_pts):
    #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    pts = np.array(mask_pts)
    color = 255
    cv2.fillPoly(mask, [pts], color)
    masked = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = image[y:y+h, x:x+w].copy()
    print(croped.shape , image.shape)
    return croped, image

def find_points(point_list: list, points: list):
    if len(point_list)>1:
        # print("start condition satisfied")
        x=point_list[0]
        print("x: ", x,point_list)
        for i in range(len(point_list)-1):
            # print(f"{i}: value->", point_list[i])
            if i == len(point_list)-2:
                # print(f'>>>>>>>>>>>>>>>>1')
                points.append((x, point_list[i+1]))
                # print("finished")
                return
            # print(f'>>>>>>>>>>>>>>>>1')
            if point_list[i+1] - point_list[i]<3:
                # print(f'>>>>>>>>>>>>>>>>>>>> 2')
                continue
            else:
                # print("in else")
                new_points = point_list[i+2:]
                find_points(new_points, points)
                points.append((x, point_list[i]))
                return

