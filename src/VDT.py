from collections import deque

from cars import find_cars
from scipy.ndimage.measurements import label

import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from config import *
from util import *
from windows import *
from train_classifier import get_clf_and_scaler


class VDT:
    def __init__(self, data_path):
        self._all_heatmaps = deque()
        self._data_path = data_path
        self._clf, self._scaler = get_clf_and_scaler(data_path)

    def process(self, image):
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        bboxes = []
        for scale in SCALE_VALUES:
            bboxes += find_cars(image, scale, self._clf, self._scaler)
        
        heat = add_heat(heat, bboxes)
        heat = apply_threshold(heat, HEATMAP_THRESHOLD)

        self._all_heatmaps.append(heat)

        if (len(self._all_heatmaps) > HEATMAP_WINDOW_SIZE):
            self._all_heatmaps.popleft()

        sum_heat = np.zeros_like(heat)
        for h in self._all_heatmaps:
            sum_heat += h
        avg_heatmap = sum_heat/len(self._all_heatmaps)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(avg_heatmap, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        draw_img = add_stats(draw_img, labels[1])

        showimg(draw_img)

        return draw_img


if __name__ == '__main__':
    args = parse_args()
    vdt = VDT(args.data_path)

    for idx in range(1, 7):
        image = mpimg.imread('./test_images/test' + str(idx) + '.jpg')
        vdt.process(image)

    # video = VideoFileClip("project_video.mp4")
    # output_video = video.fl_image(vdf.process)
    # output_video.write_videofile("output_project_video.mp4", audio=False)        


    