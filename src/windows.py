import cv2
import numpy as np
from scipy.ndimage.measurements import label

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
