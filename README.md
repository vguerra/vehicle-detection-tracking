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
    - [`VDT.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/VDT.py): Higher level implementation of the pipeline.
    - [`windows.py`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/windows.py): All code related to drawing boxes on images.
---

### Histogram of Oriented Gradients

The [first step in the pipeline](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/VDT.py#L20) is to have a classifier and a feature scaler. This two objects help us later in the pipeline to convert the video frames to an array of suitable features our classifier can take in so that we can detect the prescence of vehicles.

To train our classifier we have two classes of labeled images. Images where vehicles appear, and image where don't. For example:

* Image labeled with Vehicle class.
<p align="center">
 <img src="https://github.com/vguerra/vehicle-detection-tracking/blob/master/output_images/vehicle.png" width="350">
</p>

* Image labeled wit No-Vehicle class.
<p align="center">
 <img src="https://github.com/vguerra/vehicle-detection-tracking/blob/master/output_images/no-vehicle.png" width="350">
</p>

So given a list of images that belong to the same class, we [extract the features](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L147-L189). And for [each image](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L90-L141) a feauture vector is computed. THe feature vector is formed of three parts:

* [Spatial Binning of Color](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L30-L42): We found useful to have included raw pixel values in the feature vector. Since including all pixels in full resolution would take a lot of space and is not necesary to capture useful information in this case, we resize the image to a size of 32x32 and flatten it.

* [Color Histogram Features](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L68-L85): For each color channel, we compute a histogram of colors using 32 bins. Regions in the images where cars appear will have similar color histograms to the ones of the cars themselves.

* [HOG features](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L45-L66): The most relevant part of the feature vector are the HOG features. For this we use the [`hog`](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L61-L66) function provided by skimage package. Before computing all features mentioned before and HOG features we first convert the image to [another color space](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L6-L27). After experimenting with different color spaces, the one that gave us better result was `YCrCb`. So the images passed to the hog function are in this color space. About the rest of the parameters for the hog function the values are:
    - `orientations`: value of 9. At this small scale we don't need a higher accuracy on the bins used for the orientation's histogram.
    - `pixels_per_cell`: value of 8. This indicates our cells are a square of 8x8 pixels.
    - `cells_per_block`: value of 2. Each block is composed of 2x2 cells.

We compute the hog features for [all 3 channels](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/features.py#L129-L135) in the image.

An visualization of a HOG image (for one channel) containing a car would be:

* Original vehicle image.
<p align="center">
 <img src="https://github.com/vguerra/vehicle-detection-tracking/blob/master/output_images/no-hog.png" width="350">
</p>

* HOG features.
<p align="center">
 <img src="https://github.com/vguerra/vehicle-detection-tracking/blob/master/output_images/hog.png" width="350">
</p>

Once all three parts of the feature vector are computed we can proceed to train our classifier.

For the training part, the following steps are followed:
* [Standarize features](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/train_classifier.py#L94-L96) by using a StandardScaler object form sklearn. This is done so that no feature dominates the objective function, which could have bad effects when learning the model parameters.
* [Shuffle and split the data set](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/train_classifier.py#L102-L104) into training data and test data. We train our model using 80% of the whole data set and compute accuracy score on the additional 20%.

As you can imagine, the whole process we just described to train our classifier is rather computational expensive. So once we have gone through this process we [cache](https://github.com/vguerra/vehicle-detection-tracking/blob/master/src/train_classifier.py#L34-L48) the classifier and the scaler to be used in future runs.

---

### Sliding Window Search

Once our classifier and scaler objects are ready. We can start processing images from the video.

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

[![Project Video](https://img.youtube.com/vi/1tYMLZGV_l8/0.jpg)](https://youtu.be/1tYMLZGV_l8)

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
