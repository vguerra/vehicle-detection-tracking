# Vehicle Detection Tracking

* [The goal](#the-goal)
* [Rubric](#rubric)
* [Writeup](#writeup)
* [The Code](#the-code)
* [Histogram of Oriented Gradients](#histogram-of-oriented-gradients)
* [Sliding Window Search](#sliding-window-search)
* [Output video](#output-video)
* [Discussion](#discussion)

---

## The Goal
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view)
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup
This document that you are reading! :).

---

### The Code
The project is organized as follows:
* [`data`](https://github.com/vguerra/vehicle-detection-tracking/tree/master/data) folder which is where the cached objects for classifier and scaler resides, as well as training data.
* [`src`](https://github.com/vguerra/vehicle-detection-tracking/tree/master/src) folder containing all source code:
    - [`cars.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/cars.py): Implementing general algorith to finds car in an image.
    - [`config.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/config.py): All configuration parameters used throught the pipeline.
    - [`features.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py): All code related to feature extraction for images.
    - [`train_classifier`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/train_classifier.py): Implementing all logic needed to train our classifier.
    - [`util.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/util.py): Utilities for debugging purposes.
    - [`windows.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/windows.py): All code related to drawing boxes on images.
---

### Histogram of Oriented Gradients

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Output Video

Now it is time to see all put together in action. Checkout this repository and at the root directory execute the following:
```
$> python src/VDT.py ./data
```
The [`output_project_video.mp4`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/output_project_video.mp4) should look something like this:

[![Project Video](https://img.youtube.com/vi/https://youtu.be/1tYMLZGV_l8/0.jpg)](https://youtu.be/https://youtu.be/1tYMLZGV_l8)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

<Heatmap examples>

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion
