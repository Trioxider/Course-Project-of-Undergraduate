import cv2
import numpy as np
from matplotlib import pyplot as plt


def template_matching(img_left, img_right, window_size, metric):
    # initialize an empty disparity map
    disparity_map_ = np.zeros(img_left.shape)
    # Traverse each pixel of the left image
    for i in range(window_size, img_left.shape[0] - window_size):
        for j in range(window_size, img_left.shape[1] - window_size):
            # Extract a window centered at (i, j) from the left image as a small patch
            patch_left = img_left[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1]
            # Find the position in the right image that best matches a small patch in the left image to achieve
            # binocular stereo vision.
            result = cv2.matchTemplate(img_right, patch_left, metric)
            # get the value and location of the maximum one and minimum one from
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # assign different match location by recognizing different metrics
            if metric == cv2.TM_SQDIFF_NORMED:
                match_loc = min_loc
            else:
                match_loc = max_loc
            # calculate disparity value of each pixel
            offset = j - match_loc[0]
            # store disparity value of each pixel
            if min_disp <= offset <= max_disp:
                # store disparity value
                disparity_map_[i, j] = offset
            else:
                # set disparity value to 0 or invalid
                disparity_map_[i, j] = 0
    return disparity_map_


def depth_from_disparity(disparity_map, focal_length, baseline):
    # Initialize an empty depth map
    depth_map = np.zeros(disparity_map.shape)
    # loop through each pixel in the disparity map
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            # get the disparity value
            disparity = abs(disparity_map[i, j])
            # avoid division by zero
            if disparity == 0:
                disparity = 0.1
            # compute the depth using the formula: depth = focal_length * baseline / disparity
            depth = np.log(focal_length * baseline / disparity) / np.log(3810) * 255
            # Assign the depth to the corresponding pixel in the depth map
            depth_map[i, j] = depth
    # Return the depth map
    return depth_map


if __name__ == '__main__':
    # Input two pictures to simulate the visual pictures obtained by binoculars
    # Remember to convert the image to grayscale
    imgL = cv2.imread('Resources/view1_1.png', 0)
    imgR = cv2.imread('Resources/view5_1.png', 0)
    min_disp = 0
    max_disp = 64
    # window_sizes = [3, 5, 7]
    # metrics = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]

    # compute disparity map
    disparity_map = template_matching(imgL, imgR, 7, cv2.TM_CCOEFF_NORMED)
    # choose focal length and baseline to compute depth map
    depth_map = depth_from_disparity(disparity_map, 1000, 100)

    # show the two maps
    plt.figure(figsize=(20, 20))
    plt.subplot(221), plt.imshow(imgR, cmap='gray')
    plt.title('Template Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(imgL, cmap='gray')
    plt.title('Origin Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(disparity_map, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(depth_map, cmap='gray')
    plt.title('Depth Map'), plt.xticks([]), plt.yticks([])
    plt.show()
