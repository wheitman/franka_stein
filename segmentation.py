import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # TODO: Get figure from the arm camera (calibrate get image)

    # Get the color image from the capture
    # ret, color_image = capture.get_color_image()
    fig, axs = plt.subplots(2,2)
    ax1, ax2, ax3, ax4 = axs.flatten()

    color_image = cv2.imread("example.png") # TODO: This is where we want the camera input

    # Convert to HSV
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    refined_image = hsv_image.copy()

    # TODO: Change based on color. Ask Will
    HUE_MIN = 170
    refined_image[refined_image[:,:,0] < 170] = [0,0,0]

    SAT_THRESHOLD = 100
    refined_image[refined_image[:,:,1] < 80] = [0,0,0]

    VAL_MIN = 180
    refined_image[refined_image[:,:,2] < 180] = [0,0,0]

    # Dilate a bit to exagerate result
    kernel = np.ones((5, 5), np.uint8) 
    refined_image = cv2.dilate(refined_image, kernel, iterations=1)

    # Show it
    ax1.set_title("Original image")
    ax1.imshow(color_image)
    ax3.set_title("Segmented image")
    ax3.imshow(refined_image)
    ax2.set_title("Saturation")
    ax2.imshow(hsv_image[:,:,1])
    ax4.set_title("Value")
    ax4.imshow(hsv_image[:,:,2])
    plt.show()