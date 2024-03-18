import cv2
import numpy as np
import pykinect_azure as pykinect
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    # pykinect.initialize_libraries()

    # Modify camera configuration
    # device_config = pykinect.default_configuration
    # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    # print(device_config)

    # Start device
    # device = pykinect.start_device(config=device_config)

    # Get capture
    # capture = device.update()

    # Get the color image from the capture
    # ret, color_image = capture.get_color_image()
    fig, axs = plt.subplots(2,2)
    ax1, ax2, ax3, ax4 = axs.flatten()
    color_image = cv2.imread("example.png")

    # Convert to RGB
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    refined_image = hsv_image.copy()

    HUE_MIN = 170
    refined_image[refined_image[:,:,0] < 170] = [0,0,0]

    SAT_THRESHOLD = 100
    refined_image[refined_image[:,:,1] < 80] = [0,0,0]

    VAL_MIN = 180
    refined_image[refined_image[:,:,2] < 180] = [0,0,0]

    # Dilate a bit to exagerate result
    kernel = np.ones((5, 5), np.uint8) 
    refined_image = cv2.dilate(refined_image, kernel, iterations=1) 

    # # Mask out red marks
    # RED_MIN_THRESHOLD = 180  # Pixels must be at least this red
    # GREEN_MAX_THRESHOLD = 150  # Pixels must be at most this green
    # BLUE_MAX_THRESHOLD = 150  # Pixels must be at most this blue
    # red_mask = np.logical_and(
    #     np.logical_and(
    #         hsv_image[:, :, 0] >= RED_MIN_THRESHOLD,
    #         hsv_image[:, :, 1] <= GREEN_MAX_THRESHOLD,
    #     ),
    #     hsv_image[:, :, 2] <= BLUE_MAX_THRESHOLD,
    # )

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

    # lower = np.array([4, 0, 7])
    # upper = np.array([87, 240, 255])
    # mask = cv2.inRange(hsv, lower, upper)

    # if not ret:
    #     continue

    # Plot the image
    # cv2.imshow("Color Image",color_image)
    # cv2.imwrite('example.png',color_image)
    # break

    # # Press q key to stop
    # if cv2.waitKey(1) == ord('q'):
    #     break
