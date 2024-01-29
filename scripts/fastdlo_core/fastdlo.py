# The following code is a V2 of the version V1 which can be found in the
# Github of the fastdlo package. That version had the same fastdlo_core
# folder, but core.py was different. Moreover, fastdlo.py was not there.
# This code is a class version of the one written by Italo, fastdlo_detection.py
# This code has not a main, but it's written as a class to be compatible
# with the process of 3D estimation of cables positions.

# IMPORT LIBRARIES
import os, cv2
from fastdlo_core.core import Pipeline

# CLASS IMPLEMENTATION OF FASTDLO DETECTOR FOR DLO3DS 
class FASTDLO():

    # Initialization of class variables
    def __init__(self, main_folder, mask_th = 127, ckpt_siam_name = "CP_similarity.pth", ckpt_seg_name = "CP_segmentation.pth", img_w = 640, img_h = 360):
        
        # Mask threshold to classify a cable: higher to avoid mistakes, lower to be flexible
        self.mask_th = mask_th

        # Insert NN param
        checkpoint_siam = os.path.join(main_folder, "fastdlo_core/checkpoints/" + ckpt_siam_name)
        checkpoint_seg = os.path.join(main_folder, "fastdlo_core/checkpoints/" + ckpt_seg_name)

        # Call the fastdlo solver pipeline
        self.fastdlo = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=img_w, img_h=img_h)

    # Take the tck value of the spline
    def generete_output_splines_msg(self, paths):
        tck_array = []
        
        # Iteration over key(_) and values (p) of splines items
        for _, p in paths.items():

            # Cardinal B-spline exploitation
            spline_extended = p["spline_extended"]
            t = spline_extended["t"]    # knots
            c = spline_extended["c"]    # control points
            k = spline_extended["k"]    # degree
            cx = c[1]
            cy = c[0]	
            tck_array.append([t, cx, cy, k])

        return tck_array

    # Main running function to classify cables
    def run(self, img, debug=False):

        # core.py run function: the inputs are the image and the threshold
        #                       the outputs are the splines dict and the binary mask
        splines, mask_output = self.fastdlo.run(img, mask_th=self.mask_th)

        # Generate the splines
        splines = self.generete_output_splines_msg(splines)
        return splines, mask_output        