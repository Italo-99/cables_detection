#!/usr/bin/env python

import sys,time,os
import cv2
import rospkg,rospy
from mBEST_core.run_detection import run_mBEST_detection

def main():

    # Detection pipeline, works only on cables of mBEST dataset
    rospy.init_node('my_node')
    rospy.loginfo("mBEST detection")
    run_mBEST = run_mBEST_detection("Test_detector")
    run_mBEST.run(sys.argv[1])

if __name__ == '__main__':

    main()