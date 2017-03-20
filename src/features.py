import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    """
    Resizes an image and returns a flattens it.

    Args:
        img: Array, Image to resize
        size: Tuple, desired size. If size = (x, y) then the feature vector will have size x * y.

    Returns:
        flattened resized image.
    """
    features = cv2.resize(img, size).ravel() 
    return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Computes HOG features for a given Image.

    Args:
        img: Array, Input image (greyscale)
        orient: Int, Number of orientation bins.
        pix_per_cell: Tuple, indicates size ( in pixels of a cell).
        cell_per_block: Tuple, indicates number of cells in a block.
        visualize: Boolean, Whether to return an image of the HOG.
        feature_vec: Boolean, Whether to call .ravel() on the result, or not.

    Returns:
        If visualize is True then returns a tuple (features, image) otherwise
        only the features vector.
    """
    return hog(img, orientations=orient, 
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block), 
        transform_sqrt=True, 
        visualise=vis,
        feature_vector=feature_vec)

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color histogram features

    Args:
        img: Array, Input image
        nbins: Int, number of equal-width bins to UserWarning
        bins_range: Tuple, lower and upper range of the bins

    Returns:
        Flattened array of histogram values for each channel.
    """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extracts features from a list of images.

    Args:
        imgs: List, list of Input Images
        color_space: String, color space to use. If not RGB, image will be converted
        spatial_size: Tupe, size of images used (spatial features). 
        hist_bins:: Int, number of histogram bins to use (color histogram features).
        orient: Int, Number of orientation bins for (HOG features).
        pix_per_cell: Tuple, size of cell in pixel (HOG feautures).
        cell_per_block: Tuple, cells per block.
        hog_channel: Int or String, Which channel of the image to use for HOG features (0, 1, 2, 'ALL')
        spatial_feat: Boolean, whether to include spatial features.
        hist_feat: Boolean, wheather to include color histogram features.
        hog_feat: Boolean, whether to include HOG features.

    Returns:
        List of image features.
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
