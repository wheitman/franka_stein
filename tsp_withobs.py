import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
# from scipy.interpolate import splprep, splev
import fast_tsp



def convert_to_binary(img, color):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lower = np.array(color) - 10
    upper = np.array(color) + 10
    mask = cv2.inRange(img, lower, upper)
    res = cv2.bitwise_and(img, img, mask = mask)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    res[res > 0] = 255
    return res


def get_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_centers(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX, cY))
    return centers

def get_index(point, centers):
    for idx, center in enumerate(centers):
        # print(center, point)
        if (center == point).all():
            return idx


def account_for_obstacles(red_points, obstacle_points, dist_arr):
    all_pairs = [(a, b) for idx, a in enumerate(red_points) for b in red_points[idx + 1:]]

    ## print pairwise distances
    # for pair in all_pairs:
    #     p1, p2 = pair
    #     print("For ", p1, " and ", p2, " the dist is ", int(np.linalg.norm(p2-p1)))

    bad_indexs = []

    for idx, pair in enumerate(all_pairs):
        p1, p2 = pair
        for obs in obstacle_points:
            p3 = obs
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            if d < 10:
                print("Obstacle between ", p1, " and ", p2)
                bad_indexs.append(idx)

                p1_index = get_index(p1, red_points)
                p2_index = get_index(p2, red_points)

                dist_arr[p1_index][p2_index] = 10000
                dist_arr[p2_index][p1_index] = 10000

    return dist_arr
    


if __name__ =="__main__":
    # print("Erase the Red, Avoid the Green")

    red = [255, 0, 0]
    green = [0, 255, 0]

    img = cv2.imread("update_red_and_green.png") ## input image

    ## blur the image
    img = cv2.GaussianBlur(img, (5,5), 0)

    cv2.imshow("Orig Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    red_mask = convert_to_binary(img, red)
    green_mask = convert_to_binary(img, green)

    ##dilate the masks
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations = 3)
    green_mask = cv2.dilate(green_mask, kernel, iterations = 3)

    # cv2.imshow("Red Mask", red_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    ## draw contours around the red mask
    red_contours = get_contours(red_mask)
    green_contours = get_contours(green_mask)
    
    red_cdraw = cv2.drawContours(img, red_contours, -1, (0,0,0), 3)

    cv2.imshow("Red Contours", red_cdraw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## find center of contours
    red_centers = get_centers(red_contours)
    green_centers = get_centers(green_contours)

    # print(red_centers)
    i = 0
    for center in red_centers:
        cX, cY = center
        print(center)
        centers_img = cv2.circle(img, (cX, cY), 5, (0,0,0), -1)
        ## write text at the center
        cv2.putText(
            img = centers_img,
            text = str(cX) + "," + str(cY) + ':' + str(i),
            org = (cX+2, cY+2),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color = (255,0,0),
            thickness = 2
        )
        i += 1
    
    cv2.imshow("Centers of Contours", centers_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Number of Red Centers: ", len(red_centers), " :which are to be erased")
    print("Number of Green Centers: ", len(green_centers), " :which are to be avoided")

    
    ## calculate pairwise distance between red centers
    red_centers = np.array(red_centers)
    red_dist = pairwise_distances(red_centers, metric = 'euclidean').astype(int)

    print(red_dist)
    print(account_for_obstacles(red_centers, green_centers, red_dist))
    # print()


    tour = fast_tsp.find_tour(red_dist)
    print(tour)

    ## draw a bouding box aroudn the individual contours
    # for contour in red_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     red_boxes = cv2.rectangle(red_contours, (x,y), (x+w, y+h), (0,0,0), 2)

    
    # cv2.imshow("Image", red_boxes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ##draw line and arrow from centers in the order of the tour
    for i in range(len(tour) - 1):
        p1 = red_centers[tour[i]]
        p2 = red_centers[tour[i+1]]
        # pathimg = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,0,0), 2)
        pathimg = cv2.arrowedLine(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,0,0), 2)

    cv2.imshow("Path", pathimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


