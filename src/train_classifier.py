from util import *
from config import *
from windows import *

from features import extract_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import time
import pickle
import os.path

def load_model(pickle_file):
    """
    Loads a Classifier.

    Args:
        pickel_file: Path to pickel file.

    Returns:
        None, if file is not found, the trained model otherwise.
    """
    if os.path.isfile(pickle_file) == False:
        return None, None
    
    f = open(pickle_file, "rb")
    from_dump = pickle.load(f)
    f.close()

    return from_dump['clf'], from_dump['scaler']

def save_model(clf, scaler, pickle_file):
    """
    Saves a Classifier and a Scaler

    Args:
        clf: Object, The classifier.
        scaler: Object, The scaler.
    """
    f = open(pickle_file, "wb")
    to_dump = {
        'clf': clf,
        'scaler': scaler
    }
    pickle.dump(to_dump, f)
    f.close()

def train_classifier(images_path):
    """
    Trains a SVM to recognice cars given training images.

    Args:
        images_path: String, path to training data.
    
    Returns:
        Tuple of trainned classifier and scaler object to be used for future predictions
    """
    car_imgs = get_images(images_path + '/vehicles/')
    non_car_imgs = get_images(images_path + '/non-vehicles/')

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

    return svc, X_scaler

def get_clf_and_scaler(data_path, pickle_file='./data/classifier.p'):
    """
    Looks for a trainned classifier in the cache,
    otherwise proceeds with trainning process.

    Args:
        data_path: String, path to cached objects and trainning data.
        pickle_file: String, path to cached objects.

    Returns:
        Tuple of trainned classifier and scaler object to be used for future predictions
    """
    clf, scaler = load_model(pickle_file)
    if clf == None:
        clf, scaler = train_classifier(data_path)
        save_model(clf, scaler, pickle_file)

    return clf, scaler


