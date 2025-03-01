import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm


@dataclass
class Args:
    data_path: Path
    """Expects data_path to contain rgb, depth, hand_masks folder with the same number of images"""

    cam_intrinsics_path: Path
    """Path to 3x3 camera intrinsics txt file"""

    out_path: Path = (
        Path(__file__).parent / "outputs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    """Path to save outputs to"""

    ignore_exceptions: bool = False
    """Whether to ignore exceptions and continue processing the next image"""


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 100)
    print(args)
    print("=" * 100)

    rgb_paths = sorted(list((args.data_path / "rgb").glob("*.jpg")))
    depth_paths = sorted(list((args.data_path / "depth").glob("*.png")))
    mask_paths = sorted(list((args.data_path / "hand_masks").glob("*.png")))
    assert len(rgb_paths) == len(depth_paths) == len(mask_paths), (
        f"{len(rgb_paths)} rgb, {len(depth_paths)} depth, {len(mask_paths)} masks"
    )
    print(f"Processing {len(rgb_paths)} images")

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

    pbar = tqdm(zip(rgb_paths, depth_paths, mask_paths))
    for rgb_path, depth_path, mask_path in pbar:
        filename = rgb_path.stem
        pbar.set_description(f"Processing {filename}")

        # Get data
        img_rgb = np.array(Image.open(rgb_path))
        img_depth = np.array(Image.open(depth_path))
        mask = np.array(Image.open(mask_path))

        if args.ignore_exceptions:
            try:
                (
                    pcd,
                    hamer_out,
                    mesh,
                    aligned_hamer_pcd,
                    finger_pts,
                    finger_pcd,
                    transformed_mesh,
                ) = process_image_with_hamer(
                    img_rgb,
                    img_depth,
                    mask,
                    camera_intrinsics,
                    detector_hamer,
                    vis=None,
                )
            except Exception as e:
                print(f"Ignoring the following exception and continuing: {e}")
                continue
        else:
            (
                pcd,
                hamer_out,
                mesh,
                aligned_hamer_pcd,
                finger_pts,
                finger_pcd,
                transformed_mesh,
            ) = process_image_with_hamer(
                img_rgb, img_depth, mask, camera_intrinsics, detector_hamer, vis=None
            )

        # Output folder
        args.out_path.mkdir(parents=True, exist_ok=True)

        # Output mesh
        transformed_mesh.export(args.out_path / f"{filename}.obj")

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
            frame_data[j] = list(finger_pts[j])
        frame_data["global_orient"] = hamer_out["global_orient"].tolist()
        with open(args.out_path / f"{filename}.json", "w") as json_file:
            json.dump(frame_data, json_file, indent=4)


if __name__ == "__main__":
    main()
