# Human Shadow

### Installation
Install the repo in your env by running this command in the top level human_shadow directory.
```
pip install -e .
```

In order to visualize point clouds, you may need to set init_renderer: bool = False in HaMeR's hamer/model/hamer.py file to ensure that HaMeR's visualizer doesn't cause errors with our open3d visualizer. 


### 3D Hand pose estimation
```
python hand_pose.py
```
This file contains 4 methods on how to obtain 3D estimates of keypoints on the hand. 

### 3D Hand pose estimation live
```
python hand_pose_live.py
```
This file will run hand pose estimation for live images. Please update the image, depth and intrinsics in it. 


### TODO
 - Move data paths to shared folder in /juno/group/human_shadow
 - Add full typing with mypy
