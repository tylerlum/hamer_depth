# HaMeR Depth

Hand pose estimation with HaMeR and RGB images, then improving the predictions with depth images.

## Installation

### HaMeR

First we install [HaMeR](https://github.com/geopavlakos/hamer) with the following instructions. Note that these instructions are very similar to the original ones, but we make a few changes:

```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

Open the `setup.py` file and comment out the following lines:
```
        # 'pytorch-lightning',
        # 'torch',
        # 'torchvision',
```

In order to visualize point clouds, you need to open `hamer/models/hamer.py` and set `init_renderer: bool = False` to ensure that HaMeR's visualizer doesn't cause errors with our open3d visualizer.

If you want to use an earlier python version (e.g., 3.8), this will mostly work, but you will need to change a few lines of code in hamer. Specifically, run `git grep "|"` then replace them manually with either (type hints: `Union[..., ...]` and `from typing import Union`) or (`{**..., **...}` for merging dicts).

```
conda create --name hamer_depth_env python=3.10
conda activate hamer_depth_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
pip install pytorch-lightning==2.0  # Compatible with torch==2.0.1, avoids making torch update
pip install numpy==1.24  # Avoid weird issue with numpy>=2
pip install -e ."[all]"
pip install -v -e third-party/ViTPose

bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano folder`.

Test that HaMeR works by running:

```
python demo.py \
    --img_folder example_data \
    --out_folder demo_out \
    --batch_size=48 \
    --side_view \
    --save_mesh \
    --full_frame
```

### This repo

Next, install this repo by running this command in the top level hamer_depth directory.

```
git clone https://github.com/tylerlum/hamer_depth.git
cd hamer_depth
pip install -e .
```

Other dependencies:
```
pip install open3d transformers trimesh rtree tyro ruff viser
```

## Running

We will use the demo data in `data/demo/`:

```
data/demo
├── cam_K.txt
├── rgb
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
├── depth
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
├── hand_mask
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
```

To sanity check your data, we have a script that will visualize the point cloud:
```
python run_viewer.py --help
usage: run_viewer.py [-h] [OPTIONS]

╭─ options ──────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                    │
│ --rgb-path PATH         Path to rgb images (required)                                      │
│ --depth-path PATH       Path to depth images (required)                                    │
│ --cam-intrinsics-path PATH                                                                 │
│                         Path to 3x3 camera intrinsics txt file (required)                  │
│ --only-idx {None}|INT   Index of image to process, only process this image (default: None) │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
```

Run it with a specific index:

```
python run_viewer.py \
--rgb-path data/demo/rgb \
--depth-path data/demo/depth \
--cam-intrinsics-path data/demo/cam_K.txt \
--only-idx 100
```


Run script help info:
```
python run.py --help
usage: run.py [-h] [OPTIONS]

╭─ options ────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                          │
│ --rgb-path PATH         Path to rgb images (required)                                            │
│ --depth-path PATH       Path to depth images (required)                                          │
│ --mask-path PATH        Path to hand masks (required)                                            │
│ --cam-intrinsics-path PATH                                                                       │
│                         Path to 3x3 camera intrinsics txt file (required)                        │
│ --out-path PATH         Path to save outputs to (default: outputs/<timestamp>)                   │
│ --hand-type {LEFT,RIGHT}                                                                         │
│                         Type of hand to process (default: RIGHT)                                 │
│ --debug, --no-debug     Whether to run in debug mode (default: False)                            │
│ --only-idx {None}|INT   Index of image to process, only process this image (default: None)       │
│ --ignore-exceptions, --no-ignore-exceptions                                                      │
│                         Whether to ignore exceptions and continue processing the next image      │
│                         (default: False)                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```


Then run:
```
python run.py \
--rgb-path data/demo/rgb \
--depth-path data/demo/depth \
--mask-path data/demo/hand_mask \
--cam-intrinsics-path data/demo/cam_K.txt \
--out-path data/demo/hand_pose_trajectory
```

This results in:
```
data/demo/hand_pose_trajectory
├── 000000.json
├── 000000.obj
├── 000000.png
├── 000001.json
├── 000001.obj
├── 000001.png
├── ...
```

Each JSON file contains 3D keypoint positions (x, y, z in meters) and global orientation:
```json
{
    "wrist_back":       [x, y, z],
    "wrist_front":      [x, y, z],
    "index_0_back":     [x, y, z],
    "index_0_front":    [x, y, z],
    "middle_0_back":    [x, y, z],
    "middle_0_front":   [x, y, z],
    "ring_0_back":      [x, y, z],
    "ring_0_front":     [x, y, z],
    "index_3":          [x, y, z],
    "middle_3":         [x, y, z],
    "ring_3":           [x, y, z],
    "thumb_3":          [x, y, z],
    "pinky_3":          [x, y, z],
    "global_orient":    [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
}
```

The script assumes the hands are right hands. If you want to process left hands, run:
```
python run.py \
--rgb-path data/demo/rgb \
--depth-path data/demo/depth \
--mask-path data/demo/hand_mask \
--cam-intrinsics-path data/demo/cam_K.txt \
--out-path data/demo/hand_pose_trajectory \
--hand-type LEFT
```

To visualize the process in debug mode for a specific frame, run:
```
python run.py \
--rgb-path data/demo/rgb \
--depth-path data/demo/depth \
--mask-path data/demo/hand_mask \
--cam-intrinsics-path data/demo/cam_K.txt \
--out-path data/demo/hand_pose_trajectory \
--debug \
--only-idx 100
```

The script may crash if no hand is detected (e.g., if the hand mask is empty because it is occluded). To run the script without crashing (will just skip the bad frames), run:

```
python run.py \
--rgb-path data/demo/rgb \
--depth-path data/demo/depth \
--mask-path data/demo/hand_mask \
--cam-intrinsics-path data/demo/cam_K.txt \
--out-path data/demo/hand_pose_trajectory \
--ignore-exceptions
```

To run the script on multiple sequences in batch (uses subprocess per sequence to avoid OOM), edit `run_multiple.py` to list your sequence directories and run:

```
python run_multiple.py
```

Each sequence directory should have the same structure as `data/demo/` (rgb/, depth/, hand_mask/, cam_K.txt). Results are saved to `hand_pose_trajectory/` within each sequence directory.

To format the code, run:

```
ruff check --extend-select I --fix .; ruff format .
```

## Running Continuously with ROS

To run the hand pose estimation continuously with ROS, first set up ROS with RoboStack (https://robostack.github.io/GettingStarted.html):

```
mamba create -n hamer_depth_ros_env python=3.11
mamba activate hamer_depth_ros_env

# this adds the conda-forge channel to the new created environment configuration
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

mamba install ros-noetic-desktop
```

Then follow the installation instructions above.

You may encounter the following error:
```
  File "/home/tylerlum/miniconda3/envs/hamer_depth_env_v3/lib/python3.11/site-packages/chumpy/ch.py", line 1203, in _depends_on
    want_out = 'out' in inspect.getargspec(func).args
                        ^^^^^^^^^^^^^^^^^^
AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
```

To fix this, modify `/home/tylerlum/miniconda3/envs/hamer_depth_env_v3/lib/python3.11/site-packages/chumpy/ch.py` and replace `getargspec` with `getfullargspec`.

Then run:
```
python hamer_depth_ros_node.py
```

Thanks to Marion Lepart and Jiaying Fang for writing most of this code!

