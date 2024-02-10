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

""" FASTDLO SERVER
This server can return a list of PoseArray() made up of splines
from detected cables using FASTDLO method""" 

# IMPORT LIBRARIES
from    cables_detection.srv    import Cables2D_Poses
from    fastdlo_core.core       import Pipeline
from    cv_bridge               import CvBridge
import  cv2
from    geometry_msgs.msg       import PoseArray,Pose
import  numpy as np
import  os
import  rospkg,rospy
from    fastdlo_detection       import process_result_core
from    sensor_msgs.msg         import Image
import  time

# CLASS IMPLEMENTATION OF FASTDLO DETECTION SERVER
class fastdlo_server:

    def __init__(self):

        # FASTDLO detector init
        rospack         = rospkg.RosPack()
        package_path    = rospack.get_path('cables_detection')
        script_path     = package_path + "/scripts/fastdlo_core/"
        ckpt_siam_name  = "CP_similarity.pth"
        ckpt_seg_name   = "CP_segmentation.pth"
        checkpoint_siam = os.path.join(script_path, "checkpoints/" + ckpt_siam_name)
        checkpoint_seg  = os.path.join(script_path, "checkpoints/" + ckpt_seg_name)
        
        # Test the following path for test
        self.fastdlo_images_path = package_path + "/scripts/figures_test/fastdlo/"  

        # Initialize ROS node
        rospy.init_node('fastdlo_server')

        # Start the cables detection server
        rospy.Service('fastdlo', Cables2D_Poses, self.splines_cables_detection)

        # Get node params
        self.IMG_H      = rospy.get_param('~image_height')
        self.IMG_W      = rospy.get_param('~image_width')
        self.step       = rospy.get_param('~steps_cable')  # points "jumped" in the spline msg, default: 10
        self.mask_th    = rospy.get_param('~mask_th')
        
        # Set cable 2D pose printer
        self.check_detection = rospy.get_param('~check_detection')

        # Call an instance of core.py pipeline fastdlo solver 
        self.p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg,
                          img_w=self.IMG_W, img_h=self.IMG_H)
        rospy.loginfo("Ready to detect cables.")

        # Run ROS spinner
        spin_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            spin_rate.sleep()

    def splines_cables_detection(self,req):

        # Measure total service computation time
        start_time_service = rospy.get_time()

        # Create the list of PoseArray
        cables = []

        # Detect cables within the current scene frame
        img = CvBridge().imgmsg_to_cv2(req.input_image,req.input_image.encoding)

        # Measure inference time
        inf_time_start = rospy.get_time()

        # Detect cables within images already saved in a folder (for tests)
        # source_img = cv2.resize(cv2.imread(self.fastdlo_images_path+
        #                         "real_images/cables.jpg", cv2.IMREAD_COLOR),(self.IMG_W, self.IMG_H))
        # splines, img_out = self.p.run(source_img=source_img, mask_th=200)

        splines, img_out = self.p.run(img,mask_th=self.mask_th)
        
        # Display inference time
        detection_time = rospy.get_time()-inf_time_start
        rospy.loginfo("Detection time: %s",detection_time)

        # Fill cables list with cables coords 
        for key, value in splines.items():
            cable = PoseArray()
            for k in range(0,len(value.get('points')),self.step):
                cable_point = Pose()
                # Fill cable 2D position with detection coords
                cable_point.position.x = value.get('points')[k][0]
                cable_point.position.y = value.get('points')[k][1]
                cable.poses.append(cable_point)
            # Add the spline to the tuple of detected cables
            cables.append(cable)
            
        # Display total service computational demand time
        rospy.loginfo("Service computation time: %s",
                      rospy.get_time()-start_time_service-detection_time)
        
        # Display results
        # cv2.imshow("input", source_img)
        # cv2.imshow("input", img)
        # cv2.imshow("output", img_out)
        # cv2.waitKey(5000)

        # Process results for test purposes
        if self.check_detection:
            rospy.loginfo("Print fastdlo result")
            process_result_core(splines,img_out,self.IMG_W,self.IMG_H)

        return [cables]
    
if __name__ == "__main__":
    
    fastdlo_server()
