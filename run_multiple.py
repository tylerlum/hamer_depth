import subprocess
from pathlib import Path

from tqdm import tqdm

from run import Args, HandType, run as run_single


def run(args: Args) -> None:
    run_with_same_process = False  # Same process seems to OOM
    if run_with_same_process:
        run_single(args)
    else:
        cmd = (
            "python run.py"
            + f" --rgb-path {args.rgb_path}"
            + f" --depth-path {args.depth_path}"
            + f" --mask-path {args.mask_path}"
            + f" --cam-intrinsics-path {args.cam_intrinsics_path}"
            + f" --out-path {args.out_path}"
            + f" --hand-type {args.hand_type}"
            + (" --debug" if args.debug else "")
            + (f" --only-idx {args.only_idx}" if args.only_idx is not None else "")
            + (" --ignore-exceptions" if args.ignore_exceptions else "")
        )
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    demo_dirs = [
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
    for demo_dir in tqdm(demo_dirs, desc="Validating demo directories"):
        assert demo_dir.exists(), f"Demo directory {demo_dir} does not exist"
        assert (demo_dir / "rgb").exists(), f"RGB directory {demo_dir / 'rgb'} does not exist"

    for demo_dir in tqdm(demo_dirs, desc="Running SAM2 for each demo directory"):
        run(
            Args(
                rgb_path=demo_dir / "rgb",
                depth_path=demo_dir / "depth",
                mask_path=demo_dir / "hand_mask",
                cam_intrinsics_path=demo_dir / "cam_K.txt",
                out_path=demo_dir / "hand_pose_trajectory",
                hand_type=HandType.LEFT,
            )
        )


if __name__ == "__main__":
    main()
