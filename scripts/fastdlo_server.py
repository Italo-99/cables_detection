#!/usr/bin/env python

""" FASTDLO SERVER
The following code has the only aim to return the
spline of a cable detected.
"""

# IMPORT LIBRARIES
from    cables_detection.srv    import Cables2D_Poses
from    cv_bridge               import CvBridge
import  cv2
from    geometry_msgs.msg       import PoseArray,Pose
import  numpy as np
import  os
import  rospkg,rospy
from    sensor_msgs.msg         import Image

# CLASS IMPLEMENTATION OF FASTDLO DETECTION SERVER
class fastdlo_server:

    def __init__(self):

        # FASTDLO detector init
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('cables_detection')
        script_path = package_path + "/scripts/fastdlo_core/"
        ckpt_siam_name = "CP_similarity.pth"
        ckpt_seg_name = "CP_segmentation.pth"
        checkpoint_siam = os.path.join(script_path, "checkpoints/" + ckpt_siam_name)
        checkpoint_seg = os.path.join(script_path, "checkpoints/" + ckpt_seg_name)
        IMG_W = 640
        IMG_H = 480

        # Initialize ROS node
        rospy.init_node('fastdlo_server')
        rospy.Service('fastdlo', Cables2D_Poses, self.splines_cables_detection)
        rospy.loginfo("Ready to detect cables.")

        # Run ROS spinner
        spin_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            spin_rate.sleep()

    def fiducial_vertices_callback(self,msg):

        self.image = msg

    def handle_centroid_aruco(self,req):

        centroids = PoseArray()

        for fiducial in self.image.fiducials:

            xc = (fiducial.x0+fiducial.x1+fiducial.x2+fiducial.x3)/4
            yc = (fiducial.y0+fiducial.y1+fiducial.y2+fiducial.y3)/4
            new_point = Pose()
            new_point.position.x = xc
            new_point.position.y = yc
            centroids.poses.append(new_point)
            fiducial_id = fiducial.fiducial_id
            rospy.loginfo("Fiducial ID: %d\nCentroid: %s", fiducial_id, centroids.poses[-1].position)

        return centroids
    
if __name__ == "__main__":
    
    fastdlo_server()
