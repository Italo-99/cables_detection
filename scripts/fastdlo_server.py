#!/usr/bin/env python

""" FASTDLO SERVER
The following code has the only aim to return the
spline of a cable detected.
"""

# IMPORT LIBRARIES
from    cables_detection.srv    import Cables2D_Poses
from    fastdlo_core.core       import Pipeline
from    cv_bridge               import CvBridge
import  cv2
from    geometry_msgs.msg       import PoseArray,Pose
# import  matplotlib.pyplot as plt
import  numpy as np
import  os
import  rospkg,rospy
# from    scipy.interpolate import splev, splprep, splrep
from    sensor_msgs.msg         import Image
import  time

""" FASTDLO SERVER
This server can return a list of PoseArray() made up of splines
from detected cables using FASTDLO method""" 

# CLASS IMPLEMENTATION OF FASTDLO DETECTION SERVER
class fastdlo_server:

    def __init__(self):

        # FASTDLO detector init
        rospack         = rospkg.RosPack()
        package_path    = rospack.get_path('cables_detection')
        script_path     = package_path + "/scripts/fastdlo_core/"
        # self.fastdlo_images_path = package_path + "/scripts/figures_test/fastdlo/"  
        ckpt_siam_name  = "CP_similarity.pth"
        ckpt_seg_name   = "CP_segmentation.pth"
        checkpoint_siam = os.path.join(script_path, "checkpoints/" + ckpt_siam_name)
        checkpoint_seg  = os.path.join(script_path, "checkpoints/" + ckpt_seg_name)
        self.IMG_W      = 640       # PARAM
        self.IMG_H      = 480       # PARAM
        self.step       = 10        # PARAM: how many points are "jumped" during msg conversion

        # Call an instance of core.py pipeline solver 
        self.p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg,
                          img_w=self.IMG_W, img_h=self.IMG_H)
        
        # Initialize ROS node
        rospy.init_node('fastdlo_server')
        rospy.Service('fastdlo', Cables2D_Poses, self.splines_cables_detection)
        rospy.loginfo("Ready to detect cables.")

        # Run ROS spinner
        spin_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            spin_rate.sleep()

    def splines_cables_detection(self,req):

        # Create the list of PoseArray
        cables = []

        # Detect cables within the current scene frame
        img = CvBridge().imgmsg_to_cv2(req.input_image,req.input_image.encoding)

        # Measure inference time
        time_start = time.time()

        # # Detect cables within images already saved in a folder
        # source_img = cv2.resize(cv2.imread(self.fastdlo_images_path+
        #                         "real_images/cables.jpg", cv2.IMREAD_COLOR),(self.IMG_W, self.IMG_H))
        # splines, img_out = self.p.run(source_img=source_img, mask_th=200)

        splines, img_out = self.p.run(img,mask_th=200)
        
        # Display inference time
        rospy.loginfo("Detection time: %s",time.time()-time_start)

        # Fill cables list with cables coords 
        for key, value in splines.items():
            cable = PoseArray()
            np_cable = value.get('points')
            for k in range(0,len(np_cable),self.step):
                cable_point = Pose()
                # Fill cable 2D position with detection coords
                cable_point.position.x = np_cable[k][0]
                cable_point.position.y = np_cable[k][1]
                cable.poses.append(cable_point)
            # Add the spline to the tuple of detected cables
            cables.append(cable)
        
        # Display results
        # cv2.imshow("input", source_img)
        # cv2.imshow("input", img)
        # cv2.imshow("output", img_out)
        # cv2.waitKey(1000)

        return [cables]
    
if __name__ == "__main__":
    
    fastdlo_server()
