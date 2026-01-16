from run import run, HandType
from tqdm import tqdm
from pathlib import Path

"""
/juno/u/kedia/FoundationPose/human_videos/Jan_15
├── brush
│   ├── anvil_brush
│   │   └── 20260115_231442
│   ├── lab_brush
│   │   └── 20260115_231622
│   └── red_brush
│       └── 20260115_231110
├── eraser
│   ├── amazon_eraser
│   │   └── 20260115_232955
│   ├── anvil_eraser
│   │   └── 20260115_233226
│   └── expo_eraser
│       └── 20260115_233123
├── hammer
│   ├── hammer_2
│   │   ├── clockwise
│   │   └── counter_clockwise
│   └── mallet
│       ├── clockwise
│       └── counter_clockwise
├── marker
│   ├── 040_large_marker
│   │   └── 20260115_232812
│   ├── sharpie_closed
│   │   └── 20260115_232717
│   └── staples_open
│       └── 20260115_232506
├── screwdriver
│   ├── black_screwdriver
│   │   └── 20260115_235139
│   ├── real_flat_screwdriver
│   │   └── 20260115_235042
│   └── red_screwdriver
│       └── 20260115_235241
├── spatula
│   ├── black_spatula
│   │   └── 20260115_233647
│   ├── spoon_spatula
│   │   └── 20260115_233602
│   └── wooden_spatula
│       └── 20260115_233858
"""


def main():
    # List of demo directories
    DEMO_DIRS = [
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/anvil_brush/20260115_231442/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/lab_brush/20260115_231622/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/red_brush/20260115_231110/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/amazon_eraser/20260115_232955/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/anvil_eraser/20260115_233226/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/expo_eraser/20260115_233123/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/hammer_2/clockwise/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/hammer_2/counter_clockwise/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/mallet/clockwise"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/mallet/counter_clockwise"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/040_large_marker/20260115_232812"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/sharpie_closed/20260115_232717/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/staples_open/20260115_232506/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/black_screwdriver/20260115_235139/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/real_flat_screwdriver/20260115_235042/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/red_screwdriver/20260115_235241/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/black_spatula/20260115_233647/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/spoon_spatula/20260115_233602/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/wooden_spatula/20260115_233858/"),
    ]
    # Validate
    for demo_dir in tqdm(DEMO_DIRS, desc="Validating demo directories"):
        assert demo_dir.exists(), f"Demo directory {demo_dir} does not exist"
        assert (demo_dir / "rgb").exists(), f"RGB directory {demo_dir / 'rgb'} does not exist"

    # Run SAM2 for each demo directory
    for demo_dir in tqdm(DEMO_DIRS, desc="Running SAM2 for each demo directory"):
        rgb_path = demo_dir / "rgb"
        depth_path = demo_dir / "depth"
        mask_path = demo_dir / "hand_mask"
        cam_intrinsics_path = demo_dir / "cam_K.txt"
        out_path = demo_dir.parent / "hand_pose_trajectory"
        run(
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
            cam_intrinsics_path=cam_intrinsics_path,
            out_path=out_path,
            hand_type=HandType.LEFT,
        )

if __name__ == "__main__":
    main()