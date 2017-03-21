import argparse
import glob

from config import DEBUG

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