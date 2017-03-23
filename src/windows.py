import cv2
import numpy as np
from scipy.ndimage.measurements import label

from features import single_img_features

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Performs a prediction on all windows defined for an image
    to search for cars.

    Args:
        img: Array, Input Image.
        windows: List, list of windows to inspect.
        clf: Object, classifier.
        scaler: Objec, scaler.
        color_space: String, color space to work on.
        spatial_size: Tuple.
        hist_bins: Int.
        hist_range: Tuple.
        orient: Int.
        pix_per_cell: Int.
        cell_per_block: Int.
        hog_channel: Int or String.
        spatial_feat: Boolean.
        hist_feat: Boolean.
        hog_feat: Boolean.

    Returns:
        List of windows where a car is detected.
    """

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Computes a list of window coordinates.

    Args:
        img: Array, Input Image.
        x_start_stop: Tuple, First and last x values to produce rectangles.
        y_start_stop: Tuple. First and last y values to produce rectangles.
        xy_window: Tuple. Size of the rectangles.
        xy_overlap: Tuple. Percentage value (between 0 and 1) for x and y respectively.
        
    Returns: 
        List of window coordinates.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws boxes on images

    Args:
        img: Array, Input image
        bboxes: List of Tuple of Tuple. Each element of a list defines: (left top corner, bottom right corner)
        color: 3-ple, RGB color.
        thick: Int, how thick the line should be.

    Returns:
        New image with drawn boxes.
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        draw_box(imcopy, bbox, color, thick)
    return imcopy

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        draw_box(img, bbox, (0, 0, 255), 6)

    return img

def draw_box(img, box, color=(0, 0, 255), thick=6):
    """
    Draws a box on an image.

    Args:
        img: Array, Input Image
        box: Tuple, defines top left and bottom right corner of the box
        color: 3-ple, RGB color of box perimeter.
        thick: Int, how thick the box boarder is.
    
    Returns:
        Image with drawn box.
    """
    cv2.rectangle(img, box[0], box[1], color, thick)

def add_heat(heatmap, bbox_list):
    """
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    """
    Thresholds values in heatmap

    Args:
        heatmap: Array, heatmap.
        threshold: Int, value used for thresholding.
    
    Returns:
        Array, thresholded array.
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap
