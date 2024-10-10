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

## Data collection 
### View ZED camera images live
 ```
 cd human_shadow/camera/
 python zed_redis_driver.py --resolution HD1080 --depth_mode NEURAL --render
```


### Collect human demonstrations 
 ```
 cd human_shadow
 python collect_human_data.py --folder demo_name -hz 30 --depth_mode NEURAL --resolution HD1080
```


## Real robot guide 

### Networking 
On the Franka Nuc, ensure that 
* PCI Ethernet is connected to "Internet"
* USB Ethernet is connected to "FR3"

### Running the franka (on franka nuc)
1. In window 1, run 
 ```
 cd marion/franka-panda.git/bin
 ./franka_panda_driver ../resources/default.yaml
```
2. In window 2, run 
 ```
 cd marion/franka-panda.git/bin
 ./franka_panda_opspace -g robotiq -a iprl
```

### Running the robotiq gripper (on franka nuc)
1. In window 1, run 
 ```
 cd franka/robotiq-gripper.git/bin
 ./robotiq_gripper_driver /dev/ttyUSB0
```

### Camera calibration
#### Simulation check (on bohg-franka)
1. In window 1, run  
 ```
 conda activate xembodiment
 cd redis-gl
./server.py
```
2. In window 2, run 
 ```
 conda activate xembodiment
 cd franka-panda/bin
./franka_panda_opspace --sim -g robotiq
```
3. In window 3, run 
 ```
 conda activate human_shadow
 cd human_shadow/camera
python collect_calibration_data.py --name test --sim --resolution HD2K
```

#### Real data collection 
1. Run the franka (see instructions above)
2. Run the robotiq gripper (see instructions above)
3. On bohg-franka, run
 ```
 conda activate human_shadow
 cd human_shadow/camera
python collect_calibration_data.py --name test --resolution HD2K
```

### TODO
 - Move data paths to shared folder in /juno/group/human_shadow
 - Add full typing with mypy
