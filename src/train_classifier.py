from util import *
from config import *

from features import extract_features

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import time

CAR_CLASS = 1
NON_CAR_CLASS = 0

if __name__ == '__main__':
    args = parse_args()

    # image paths
    root_path = args.data_path
    car_imgs = get_images(root_path + '/vehicles/')
    non_car_imgs = get_images(root_path + '/non-vehicles/')

    print('Computing car features')
    car_features = extract_features(car_imgs,
        color_space=COLOR_SPACE,
        spatial_size=SPATIAL_SIZE,
        hist_bins=HIST_BINS,
        orient=ORIENT,
        pix_per_cell=PIX_PER_CELL,
        cell_per_block=CELL_PER_BLOCK,
        hog_channel=HOG_CHANNEL,
        spatial_feat=SPATIAL_FEAT,
        hist_feat=HIST_FEAT,
        hog_feat=HOG_FEAT)
    print(len(car_features))

    print('Computing non-car features')
    non_car_features = extract_features(non_car_imgs,
        color_space=COLOR_SPACE,
        spatial_size=SPATIAL_SIZE,
        hist_bins=HIST_BINS,
        orient=ORIENT,
        pix_per_cell=PIX_PER_CELL,
        cell_per_block=CELL_PER_BLOCK,
        hog_channel=HOG_CHANNEL,
        spatial_feat=SPATIAL_FEAT,
        hist_feat=HIST_FEAT,
        hog_feat=HOG_FEAT)
    print(len(non_car_features))
    
    X = np.vstack((car_features, non_car_features)).astype(np.float64)                        
    print('X shape: {}'.format(X.shape))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

        