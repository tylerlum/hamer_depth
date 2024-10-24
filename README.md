# Human Shadow

## Installation
Install the repo in your env by running this command in the top level human_shadow directory.
```
pip install -e .
```

In order to visualize point clouds, you may need to set init_renderer: bool = False in HaMeR's hamer/model/hamer.py file to ensure that HaMeR's visualizer doesn't cause errors with our open3d visualizer. 



## Video processing
1. Process every video to extract hand segmentation masks and poses:
To process every video in /juno/group/shared/raw_data/data_jiaying_1
```
python process_human_data.py --demo_name data_jiaying_1 --use_shared
```

To process every video in human_shadow/human_shadow/data/videos/data_jiaying_1
```
python process_human_data.py --demo_name data_jiaying_1
```

2. Extract the segmentation masks of the robot
```
python generate_robot_seg_overlay.py --demo_name data_jiaying_1 --use_shared
```
or
```
python generate_robot_seg_overlay.py --demo_name data_jiaying_1 
```


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

## RedisGL Franka + HaMeR visualization 
On Franka NUC:
1. Run 
 ```
 cd marion/franka-panda.git/bin
 ./franka_panda_driver ../resources/default.yaml
```

On bohg-franka:
1. In window 1, run 
 ```
 cd redis-gl
 ./server.py
```

2. In window 2, run 
 ```
 cd franka-panda/bin/
 ./franka_panda_opspace -g robotiq --mirror --robot_host 172.24.68.230 --robot_pass iprl
```

3. In window 3, run
 ```
 cd human_shadow/human_shadow/camera/
 python zed_redis_driver.py --resolution HD1080 --depth_mode NEURAL
```

4. In window 4, run 
 ```
cd human_shadow/human_shadow/
python visualize_hand_redis.py --render
```

5. In chrome, go to http://localhost:8000/simulator.html




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
