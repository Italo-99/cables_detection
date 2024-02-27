#!/usr/bin/env python

"""
*
 * Software License Agreement (Apache Licence 2.0)
 *
 *  Copyright (c) [2024], [Andrea Pupa] [italo Almirante]
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3. The name of the author may not be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: [Andrea Pupa] [Italo Almirante]
 *  Created on: [2024-01-17]
*
"""

""" 
CITATIONS

@ARTICLE{9830852,
  author={Caporali, Alessio and Galassi, Kevin and Zanella, Riccardo and Palli, Gianluca},
  journal={IEEE Robotics and Automation Letters}, 
  title={FASTDLO: Fast Deformable Linear Objects Instance Segmentation}, 
  year={2022},
  volume={7},
  number={4},
  pages={9075-9082},
  doi={10.1109/LRA.2022.3189791}}
  """

# IMPORT LIBRARIES
import os, cv2, time, sys, numpy as np
from fastdlo_core.core import Pipeline
from fastdlo_core.fastdlo import FASTDLO
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep, splrep
import rospy,rospkg

# Generate an (x,y) spline from tck B-curve values
def xyspline_from_tck(tck):

    # tck is a list of [np.array(N), np.array(N+4,2), int]
    #   t: knots          -> list of floats from 0 to 1 of size N
    #   c: control points -> np.array of two elements, each of size N+4
    #   k: degree         -> int, degree of the spline 
    # xyspline is a np.array(nx2)

    # Adjust the tck values if cx and cy are in a different array
    if(len(tck) == 4):
        t,cx,cy,k = tck
        c = [cx,cy]
        tck = [t,c,k]

    spline_points= splev(tck[0],tck,der=0) # XY values of the spline as two np.arrays
    
    # xd_, yd_   = splev(tck[0], tck, der=1)
    # xdd_, ydd_ = splev(tck[0], tck, der=2)

    return spline_points,tck

# Generate tck B-curve values from (x,y) spline
def tck_from_xyspline(spline):

    # tck is a list of [np.array(N), np.array(N+4,2), int]
    #   t: list of floats from 0 to 1 of size N
    #   c: np.array of two elements, each of size N+4
    #   k: int, degree of the spline 
    # xyspline is a numpy nd array of size [Nx2]

    # Fit a B-spline to the given data
    tck, u = splprep(spline.T, u=None, k=3, s=0.0, per=0)
    return tck

# Plot a 2d spline given knots, control points and spline degree
def plot_2d_spline(tck,color,IMG_H):

    spline_points,tck = xyspline_from_tck(tck)  # Transform tck values into xy coords

    # Get control points (as list) and degree data to plot the results
    c = tck[1]
    k = tck[2]
    c_p = []
    for j in range(len(c[0])):
        c_p.append([c[0][j],c[1][j]])

    # Plot the control points
    # plt.scatter(*zip(*c_p), color='red', label='Control Points',marker='o')

    # Plot the spline curve
    plt.plot(spline_points[0],IMG_H-spline_points[1],label=f'Spline (k={k})',color=color)

    # Set labels and legend
    plt.xlabel('Image width  - w')
    plt.ylabel('Image height - h')
    plt.legend()

    # Show the plot
    # plt.show()

# Process data coming from spline of core.py (overall dict, spline with main coordinates AND tck)
def process_result_core(splines,mask_out,IMG_W,IMG_H,source_img):

    # Get splines image from spline items to verify xy points are pixels coordinates
    cables = []
    for key, value in splines.items():                      # Iterate over the cables
        cables.append(np.int64(value.get('points')))        # Get cables 2D pixels coordinates
    spline_img = np.zeros((IMG_H,IMG_W,3), dtype=int)       # Convert spline to image

    # Store some colors in the image if that coordinate is in the cables array of np.arrays
    colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],
              [0.5,0.5,0.5],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]

    for k in range(min(len(cables),len(colors))):
        for x, y in cables[k]:
            spline_img[x,y,0] = colors[k][0]
            spline_img[x,y,1] = colors[k][1]
            spline_img[x,y,2] = colors[k][2]

    spline_img = np.uint8(spline_img*255)

    # Colored source image, but by giving different colors to each cable
    # canvas = source_img.copy()
    # canvas = cv2.addWeighted(canvas, 1.0, mask_out, 0.8, 0.0)
    
    # Show the results
    cv2.imshow("input", source_img)
    # cv2.imshow("canvas", canvas)
    cv2.imshow("output", mask_out)
    cv2.imshow("mySpline",spline_img)
    cv2.waitKey(5000)                  # Wait 5 seconds than close
    cv2.destroyAllWindows()

    # Compute tck B-curve values from xy spline (to show that they are similar to
    #   tck values already in the cables array)
    colors = ['red','blue','green']
    index = 0

    # Iterate over cables
    for key, value in splines.items():
        spline_yx = value.get('points')

        # Exchange values between x and y position   
        splinexy = np.zeros((len(spline_yx),2))
        for k in range(len(spline_yx)):
            splinexy[k][0] = spline_yx[k][1]
            splinexy[k][1] = spline_yx[k][0]

        # Compute tck values and print
        tck = tck_from_xyspline(splinexy)  # Pass (x,y) spline values as ndarray
        plot_2d_spline(tck,colors[index],IMG_H)
        index += 1

    # Show the curve computed
    plt.tight_layout()
    plt.show()                  # The user can close the plot from the gui 

# Process data coming from spline of fastdlo.py (tck extended spline, only spline interpolation)
def process_result_fastdlo(splines,mask_out):

    # Input: splines (list) -> t:  nd.array of size N 
    #                          cx: x coords of control points
    #                          cy: y coords of control points
    #                          k:  spline degree
    #        mask_out       -> binary (colored) mask of cable segmentation

    # Colored source image, but by giving different colors to each cable
    canvas = source_img.copy()
    canvas = cv2.addWeighted(canvas, 1.0, mask_out, 0.8, 0.0)

    # Print 2D reconstructed splines
    colors = ['blue','green','yellow']
    index = 0
    for cable in splines:
        plot_2d_spline(cable,colors[index])
        index += 1

    plt.tight_layout()
    plt.show()              # The user can close the plot from the gui 

    # Show the results
    cv2.imshow("input", source_img)
    cv2.imshow("canvas", canvas)
    cv2.imshow("output", mask_out)
    cv2.waitKey(5000)                  # Wait 5 seconds than close all windows
    cv2.destroyAllWindows() 

    # Close the window
    rospy.signal_shutdown("Shutting down fastdlo detector node")
    
# CUSTOM MAIN FUNCTION: made by Italo to try understand the way this library works
if __name__ == "__main__":

    # Node initialization
    rospy.init_node('my_node')
    rospy.loginfo("fastdlo detection")

    # SOURCE DATA
    ######################
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('cables_detection')
    images_path = package_path + "/scripts/figures_test/fastdlo/"
    script_path = package_path + "/scripts/fastdlo_core/"
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"
    checkpoint_siam = os.path.join(script_path, "checkpoints/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "checkpoints/" + ckpt_seg_name)
    IMG_W = 640
    IMG_H = 360
    IMG_PATH = os.path.join(images_path,"real_images/cables.jpg")
    # IMG_PATH = os.path.join(images_path,"test_images/0.jpg")
    ######################

    # CORE.PY PIPELINE
    
    # Call an instance of core.py pipeline solver 
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

    # Take the source color image
    source_img = cv2.resize(cv2.imread(IMG_PATH, cv2.IMREAD_COLOR),(IMG_W, IMG_H))
    
    # Measure prediction time
    time_start = time.time()

    # Call the solver from core.py pipeline class
    spline, img_out = p.run(source_img=source_img, mask_th=200)

    # Prediction time results
    print("Detection time: {}".format(time.time()-time_start))

    # Process and display the results
    process_result_core(spline, img_out, IMG_W, IMG_H)    # key:{points,der,der2,radius,splineextended(t,c,k)}

    # FASTDLO.PY PIPELINE

    # Another way to make it work is to call fastdlo.py,the same class called by dlo3ds pipeline
    mainfolder = package_path + "/scripts"
    
    # Call an instance of fastdlo solver 
    f = FASTDLO(main_folder = mainfolder, mask_th = 200,img_w = IMG_W,img_h = IMG_H)

    # Measure prediction time
    time_start = time.time()

    # Call the solver with fastdlo.py fastdlo library
    splines_f, mask_output_f = f.run(img=source_img)

    # Prediction time results
    print("Detection time: {}".format(time.time()-time_start))

    # Display and process result
    process_result_fastdlo(splines_f, mask_output_f)    # array[N_cables] -> [t,cx,cy,k]