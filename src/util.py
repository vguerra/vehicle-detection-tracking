import argparse
import glob

from config import DEBUG

import cv2

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (255, 255, 255)
FONT_SIZE = 2.0


def get_images(path, pattern="*.png"):
    return glob.glob(path + '/**/' + pattern, recursive=True)

def showimg(img, title="image"):
    """
    Show an image if DEBUG is activated

    Args:
        img: Array, Input image to showimg
        title: String, Title to show on the image

    Returns:
        Nothing
    """
    if DEBUG:
        cv2.imshow('{} - dtype: {}'.format(title, img.dtype), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def saveimg(img, title="image", ext="png"):
    """
    Save image to file system if DEBUG is activated

    Args:
        img: Array, Input image to write in file system.
        title: String, file name to use to save the image.
    
    Returns:
        Nothing
    """
    if DEBUG:
        cv2.imwrite('./output_images/' + title + '.' +  ext, img)

def add_stats(img, cars, alpha=0.7):
    """
    Add a semi-transparent region on top of image and count of cars
    on top of it.

    Args:
        img: Array, Input Image.
        alpha: Double, weight of overlay.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 120), (135, 206, 250), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    txt = "Vehicle's count: {0}".format(cars)
    cv2.putText(img, txt, (900, 70), FONT, FONT_SIZE, FONT_COLOR, 1)

    return img

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help="Path to directory containing training images")
    # parser.add_argument('--video', help="Input video")
    # parser.add_argument('--image', help="Input image")
    # parser.add_argument('--output', help="output video file name")
    return parser.parse_args()

