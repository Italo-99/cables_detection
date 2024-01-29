#!/usr/bin/env python

import sys,time
import cv2
from mBEST_core.mBEST import mBEST
from mBEST_core.mBEST.color_masks import detect_pink_green_teal

class run_mBEST_detection:
    
    def __init__(self, name):
        
        self.name = name

    def run(self,img_path):

        # Call the class of cables segmentation
        dlo_seg = mBEST()

        # Measure inference time
        start_time = time.time()

        # Read the image
        img = cv2.imread(img_path)

        # Create MASK -> e grazie al cazzo
        mask = detect_pink_green_teal(img)

        # Call the solver
        dlo_seg.set_image(img)
        paths, path_img = dlo_seg.run(mask, plot=True)

        # dlo_seg.run(mask, intersection_color=[255, 0, 0], plot=True)
        # dlo_seg.run(mask, save_fig=True, save_path="", save_id=0)

        # Compute inference time
        print("Inference time: {} s".format(time.time()-start_time))

# MAIN mBESt detection pipeline
if __name__ == "__main__":

    run_mBEST = run_mBEST_detection("Test_detector")
    run_mBEST.run(sys.argv[1])