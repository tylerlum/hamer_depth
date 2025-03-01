# HaMeR Depth

Hand pose estimation with HaMeR and RGB images, then improving the predictions with depth images.

## Installation

First we install [HaMeR](https://github.com/geopavlakos/hamer). We copy their instructions here:

```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
conda create --name hamer_depth_env python=3.10
conda activate hamer_depth_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
pip install pytorch-lightning==2.0  # Compatible with torch==2.0.1, avoids making torch update
pip install numpy==1.24  # Avoid weird issue with numpy>=2
pip install -e .[all]
pip install -v -e third-party/ViTPose

bash fetch_demo_data.sh
```

Note: If you want to use an earlier python version (e.g., 3.8), this will mostly work, but you will need to change a few lines of code in hamer. Specifically, run `git grep "|"` then replace them manually with either (type hints: `Union[..., ...]` and `from typing import Union`) or (`{**..., **...}` for merging dicts).

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano folder`.

Test that HaMeR works by running:

```
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

Next, install this repo by running this command in the top level hamer_depth directory.

```
git clone https://github.com/tylerlum/hamer_depth.git
cd hamer_depth
pip install -e .
```

Other dependencies:
```
pip install open3d mediapy transformers trimesh rtree tyro
```

In order to visualize point clouds, you may need to set init_renderer: bool = False in HaMeR's hamer/model/hamer.py file to ensure that HaMeR's visualizer doesn't cause errors with our open3d visualizer. 

```
python run.py TODO
```

Thanks to Marion Lepart and Jiaying Fang for writing most of this code!