from pathlib import Path

import SimpleITK as sitk

from convert_dirlab_common import (
    build_arg_parser,
    ensure_case_folders,
    extract_archive,
    flip_landmarks_z,
    segment_with_totalsegmentator,
    parse_meta,
    read_landmarks,
    read_raw_image,
    resolve_output_folder,
    save_metadata,
    temporary_directory,
    write_landmarks,
)


RAW_META = {
    1: """Image Dims: 256 x 256 x 94
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 1280
Displacement (mm): 4.01 (2.91)
Repeats (#/#): 200/3
Observers (mm): 0.85 (1.24)
Lowest Error (mm): Observer Uncertainty Threshold""",
    2: """Image Dims: 256 x 256 x 112
Voxels (mm): 1.16 x 1.16 x 2.5
Features (#): 1487
Displacement (mm): 4.65 (4.09)
Repeats (#/#): 200/3
Observers (mm): 0.70 (0.99)
Lowest Error (mm): 0.72 (0.87)""",
    3: """Image Dims: 256 x 256 x 104
Voxels (mm): 1.15 x 1.15 x 2.5
Features (#): 1561
Displacement (mm): 6.73 (4.21)
Repeats (#/#): 200/3
Observers (mm): 0.77 (1.01)
Lowest Error (mm): 0.90 (1.05)""",
    4: """Image Dims: 256 x 256 x 99
Voxels (mm): 1.13 x 1.13 x 2.5
Features (#): 1166
Displacement (mm): 9.42 (4.81)
Repeats (#/#): 200/3
Observers (mm): 1.13 (1.27)
Lowest Error (mm): 1.21 (1.19)""",
    5: """Image Dims: 256 x 256 x 106
Voxels (mm): 1.10 x 1.10 x 2.5
Features (#): 1268
Displacement (mm): 7.10 (5.14)
Repeats (#/#): 200/3
Observers (mm): 0.92 (1.16)
Lowest Error (mm): 1.07 (1.46)""",
    6: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 419
Displacement (mm): 11.10 (6.98)
Repeats (#/#): 150/3
Observers (mm): 0.97 (1.38)
Lowest Error (mm): Observer Uncertainty Threshold""",
    7: """Image Dims: 512 x 512 x 136
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 398
Displacement (mm): 11.59 (7.87)
Repeats (#/#): 150/3
Observers (mm): 0.81 (1.32)
Lowest Error (mm): Observer Uncertainty Threshold""",
    8: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 476
Displacement (mm): 15.16 (9.11)
Repeats (#/#): 150/3
Observers (mm): 1.03 (2.19)
Lowest Error (mm): Observer Uncertainty Threshold""",
    9: """Image Dims: 512 x 512 x 128
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 342
Displacement (mm): 7.82 (3.99)
Repeats (#/#): 150/3
Observers (mm): 0.75 (1.09)
Lowest Error (mm): 0.91 (0.93)""",
    10: """Image Dims: 512 x 512 x 120
Voxels (mm): 0.97 x 0.97 x 2.5
Features (#): 435
Displacement (mm): 7.63 (6.54)
Repeats (#/#): 150/3
Observers (mm): 0.86 (1.45)
Lowest Error (mm): Observer Uncertainty Threshold""",
}


def _read_meta(case_id: int) -> dict:
    return parse_meta(RAW_META[case_id])


def _case_input_folder(tmp_dir: Path, case_id: int) -> Path:
    if case_id == 8:
        return tmp_dir / f"Case{case_id}Deploy"
    return tmp_dir / f"Case{case_id}Pack"


def _landmarks_input(case_folder: Path, case_id: int) -> tuple[Path, str]:
    if case_id <= 5:
        return case_folder / "ExtremePhases", f"Case{case_id}_300"
    return case_folder / "extremePhases", f"case{case_id}_dirLab300"


def _phase_image_path(case_folder: Path, case_id: int, phase: int) -> Path:
    candidates = (
        f"case{case_id}_T{phase * 10:02d}.img",
        f"case{case_id}_T{phase * 10:02d}_s.img",
        f"case{case_id}_T{phase * 10:02d}-ssm.img",
    )
    for candidate in candidates:
        image_path = case_folder / "Images" / candidate
        if image_path.exists():
            return image_path
    raise FileNotFoundError(
        f"Could not find phase {phase:02d} image for case {case_id}"
    )


def convert() -> None:
    parser = build_arg_parser("Convert raw DIR-Lab 4DCT data into INR-DIR format.")
    args = parser.parse_args()

    output_folder = resolve_output_folder(args.input_folder, args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with temporary_directory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for case_id in range(1, 11):
            archive_path = args.input_folder / f"Case{case_id}Pack.zip"
            print(f"Converting 4DCT case {case_id:02d} from {archive_path.name}")
            extract_archive(archive_path, tmp_dir)

            input_case_folder = _case_input_folder(tmp_dir, case_id)
            output_case_folder = output_folder / f"case_{case_id:02d}"
            folders = ensure_case_folders(output_case_folder)
            meta = _read_meta(case_id)

            landmarks_folder, base_name = _landmarks_input(input_case_folder, case_id)
            fixed_landmarks = flip_landmarks_z(
                read_landmarks(landmarks_folder / f"{base_name}_T50_xyz.txt"),
                meta,
            )
            moving_landmarks = flip_landmarks_z(
                read_landmarks(landmarks_folder / f"{base_name}_T00_xyz.txt"),
                meta,
            )
            write_landmarks(
                moving_landmarks,
                folders["landmarks_raw"] / "extreme_landmarks_0.csv",
            )
            write_landmarks(
                fixed_landmarks,
                folders["landmarks_raw"] / "extreme_landmarks_5.csv",
            )

            for phase in range(10):
                print(f"  Writing phase {phase:02d}")
                image = read_raw_image(
                    _phase_image_path(input_case_folder, case_id, phase), meta
                )
                image_path = folders["images"] / f"phase_{phase:02d}.nii"
                sitk.WriteImage(image, str(image_path))

                if args.use_totalsegmentator:
                    segment_with_totalsegmentator(
                        image_path=image_path,
                        segmentation_path=output_case_folder
                        / f"segmentations_{phase:02d}.nii",
                        device=args.totalsegmentator_device,
                    )
                    segment_with_totalsegmentator(
                        image_path=image_path,
                        segmentation_path=output_case_folder
                        / f"segmentations_{phase:02d}",
                        device=args.totalsegmentator_device,
                        task="body",
                        multilabel=False,
                    )
                    segment_with_totalsegmentator(
                        image_path=image_path,
                        segmentation_path=output_case_folder
                        / "lung_vessels"
                        / f"lung_vessels_{phase:02d}.nii",
                        device=args.totalsegmentator_device,
                        task="lung_vessels_LEGACY",
                        probabilities_path=output_case_folder
                        / "lung_vessels"
                        / f"lung_vessels_prob_{phase:02d}.npz",
                    )

            save_metadata(output_case_folder, meta)


if __name__ == "__main__":
    convert()
