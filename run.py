import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

from hamer_depth.utils.hand_type import HandType


@dataclass
class Args:
    rgb_path: Path
    """Path to rgb images"""

    depth_path: Path
    """Path to depth images"""

    mask_path: Path
    """Path to hand masks"""

    cam_intrinsics_path: Path
    """Path to 3x3 camera intrinsics txt file"""

    out_path: Path = (
        Path(__file__).parent / "outputs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    """Path to save outputs to"""

    hand_type: HandType = HandType.RIGHT
    """Type of hand to process"""

    debug: bool = False
    """Whether to run in debug mode"""

    only_idx: Optional[int] = None
    """Index of image to process, only process this image"""

    ignore_exceptions: bool = False
    """Whether to ignore exceptions and continue processing the next image"""


def convert_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    # depth is either in meters or millimeters
    # Need to convert to meters
    # If the max value is greater than 100, then it's likely in mm
    in_mm = depth.max() > 100
    if in_mm:
        return depth / 1000
    else:
        return depth


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 100)
    print(args)
    print("=" * 100)

    rgb_paths = sorted(list((args.rgb_path).glob("*.jpg")))
    depth_paths = sorted(list((args.depth_path).glob("*.png")))
    mask_paths = sorted(list((args.mask_path).glob("*.png")))
    assert len(rgb_paths) == len(depth_paths) == len(mask_paths), (
        f"{len(rgb_paths)} rgb, {len(depth_paths)} depth, {len(mask_paths)} masks"
    )
    num_images = len(rgb_paths)
    print(f"Processing {num_images} images")

    # Put these imports here to avoid heavy import at the top
    # As this makes --help slow and ugly
    from hamer_depth.detectors.detector_hamer import (
        DetectorHamer,
    )
    from hamer_depth.utils.run_utils import (
        convert_intrinsics_matrix_to_dict,
        get_camera_matrix_from_file,
        process_image_with_hamer,
    )

    detector_hamer = DetectorHamer()

    # Get intrinsics
    camera_matrix = get_camera_matrix_from_file(args.cam_intrinsics_path)
    camera_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)

    pbar = tqdm(
        enumerate(zip(rgb_paths, depth_paths, mask_paths)),
        total=num_images,
        dynamic_ncols=True,
    )
    for i, (rgb_path, depth_path, mask_path) in pbar:
        if args.only_idx is not None and i != args.only_idx:
            continue

        filename = rgb_path.stem
        pbar.set_description(f"Processing {filename}")

        # Get data
        img_rgb = np.array(Image.open(rgb_path))
        img_depth = np.array(Image.open(depth_path))
        mask = np.array(Image.open(mask_path))

        # Convert depth to meters
        img_depth = convert_depth_to_meters(img_depth)

        if args.ignore_exceptions:
            try:
                (
                    _,
                    hamer_out,
                    _,
                    _,
                    hand_keypoints_dict,
                    _,
                    hand_mesh_accurate,
                    _,
                ) = process_image_with_hamer(
                    img_rgb=img_rgb,
                    img_depth=img_depth,
                    mask=mask,
                    cam_intrinsics=camera_intrinsics,
                    detector_hamer=detector_hamer,
                    hand_type=args.hand_type,
                    debug=args.debug,
                )
            except Exception as e:
                print(f"Ignoring the following exception and continuing: {e}")
                continue
        else:
            (
                _,
                hamer_out,
                _,
                _,
                hand_keypoints_dict,
                _,
                hand_mesh_accurate,
                _,
            ) = process_image_with_hamer(
                img_rgb=img_rgb,
                img_depth=img_depth,
                mask=mask,
                cam_intrinsics=camera_intrinsics,
                detector_hamer=detector_hamer,
                hand_type=args.hand_type,
                debug=args.debug,
            )

        # Output folder
        args.out_path.mkdir(parents=True, exist_ok=True)

        # Output mesh
        hand_mesh_accurate.export(args.out_path / f"{filename}.obj")

        # Output annotated image
        cv2.imwrite(args.out_path / f"{filename}.png", hamer_out["annotated_img"])

        # Output json
        joint_poses = hamer_out["kpts_3d"]
        assert joint_poses.shape == (21, 3), f"{joint_poses.shape} != (21, 3)"
        joint_names = [
            "wrist_back",
            "wrist_front",
            "index_0_back",
            "index_0_front",
            "middle_0_back",
            "middle_0_front",
            "ring_0_back",
            "ring_0_front",
            "index_3",
            "middle_3",
            "ring_3",
            "thumb_3",
        ]
        frame_data = {}
        for j in joint_names:
            frame_data[j] = list(hand_keypoints_dict[j])
        frame_data["global_orient"] = hamer_out["global_orient"].tolist()
        with open(args.out_path / f"{filename}.json", "w") as json_file:
            json.dump(frame_data, json_file, indent=4)


if __name__ == "__main__":
    main()
