from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm


@dataclass
class Args:
    rgb_path: Path
    """Path to rgb images"""

    depth_path: Path
    """Path to depth images"""

    cam_intrinsics_path: Path
    """Path to 3x3 camera intrinsics txt file"""

    only_idx: Optional[int] = None
    """Index of image to process, only process this image"""


def convert_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    in_mm = depth.max() > 100
    if in_mm:
        return depth / 1000
    return depth


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 100)
    print(args)
    print("=" * 100)

    rgb_paths = sorted(list((args.rgb_path).glob("*.png")))
    depth_paths = sorted(list((args.depth_path).glob("*.png")))
    assert len(rgb_paths) == len(depth_paths), (
        f"{len(rgb_paths)} rgb, {len(depth_paths)} depth"
    )
    num_images = len(rgb_paths)
    print(f"Processing {num_images} images")

    from hamer_depth.utils.cam_intrinsics_utils import (
        convert_intrinsics_matrix_to_dict,
        get_camera_matrix_from_file,
    )
    from hamer_depth.utils.pcd_utils import get_point_cloud_of_segmask
    from hamer_depth.utils.run_utils import visualize_geometries

    camera_matrix = get_camera_matrix_from_file(args.cam_intrinsics_path)
    camera_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)

    pbar = tqdm(
        enumerate(zip(rgb_paths, depth_paths)),
        total=num_images,
        dynamic_ncols=True,
    )
    for i, (rgb_path, depth_path) in pbar:
        if args.only_idx is not None and i != args.only_idx:
            continue

        filename = rgb_path.stem
        pbar.set_description(f"Processing {filename}")

        img_rgb = np.array(Image.open(rgb_path))
        img_depth = np.array(Image.open(depth_path))
        H, W, C = img_rgb.shape
        assert C == 3, f"{C} != 3"
        assert img_depth.shape == (H, W), f"{img_depth.shape} != ({H}, {W})"

        img_depth = convert_depth_to_meters(img_depth)

        full_pcd = get_point_cloud_of_segmask(
            mask=np.ones_like(img_depth),
            depth_img=img_depth,
            img=img_rgb,
            intrinsics=camera_intrinsics,
            visualize=False,
        )

        visualize_geometries(
            width=img_rgb.shape[1],
            height=img_rgb.shape[0],
            cam_intrinsics=camera_intrinsics,
            geometries=[full_pcd],
        )


if __name__ == "__main__":
    main()
