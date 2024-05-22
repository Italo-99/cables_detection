# CABLES DETECTION
This package contains AI detection ROS1 codes to find cable shapes, carried out as research fellow activity Italo Almirante within the Ars Control Laboratory, with the collaboration of the professor Cristian Secchi and the post-doctoral researcher Andrea Pupa.

# Package guide

This package is fully compatible with ROS Noetic distro, within Ubuntu 20.04.

The folder "fastdlo_core" is a detection package to find cables within a RGB image. It has not been developed by the mainteners of this repository, but by a researchers' group of UNIBO, Italy.
Citation: https://github.com/lar-unibo/fastdlo.git.

The code "scripts/fastdlo_detection.py" has been written by the mainteiners of this repo, to ease detection interface with other projects.

The code "scripts/fastdlo_server.py" is a ROS server which works to receive an image from an external client node and returns the detected cable according to the type in the file "srv/Cables2D_Poses.srv".

The code "launch/fastdlo_detector_server.launch" is a ROS launch file which calls the cable detector server ("scripts/fastdlo_server.py") with some parameters, such as the image pixels size, the discretization step of the array of cable points (higher for coarse approximation, lower for fine), the confidence threshold to classify a pixel as cable-belonging (in [0,255]), and the visualization check enabler. 

# Installation guide

Follow the guide for the dependencies installation of the "FASTDLO" package cited above. Don't clone that repository, since it is already included in this package.

Furthermore, the command below has to be run on the terminal:

    pip install torch pygments openxlab numpy tzdata

Moreover, a known issue is here discribed: the package "networkx" has not been already adjusted according to the latest update of numpy. Downgrade to an older numpy version is not suggested. Instead, it's preferable to execute the following steps:

1. Go to /usr/lib/python3/dist-packages/networkx/readwrite/graphml.py, line 346, and change "np.int" with "int".

2. Go to /usr/lib/python3/dist-packages/networkx/readwrite/gexf.py, line 223, and change "np.int" with "int".

Finally, to compile and execute the package, remember to download the UNIBO trained models below, and place them in the checkpoints folder inside fastdlo_core:

    https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view

To share comments or feedbacks, contact the maintainer at this email: italo.almirante@unimore.it