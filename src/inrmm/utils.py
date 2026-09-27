from pathlib import Path
import json

import scipy.ndimage as ndi
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml


def get_bounding_box(
    mask: torch.Tensor, padding: int | tuple[int, ...] = 0
) -> tuple[slice, ...]:
    """Return the padded nonzero bounding box of an N-D mask."""
    if isinstance(padding, int):
        padding = (padding,) * mask.ndim
    if len(padding) != mask.ndim:
        raise ValueError("Padding dimensionality must match the mask")
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    return tuple(
        slice(
            max(int(nonzero[:, axis].min()) - padding[axis], 0),
            min(int(nonzero[:, axis].max()) + padding[axis] + 1, mask.shape[axis]),
        )
        for axis in range(mask.ndim)
    )


def pixel_coords_to_norm(pixel_coords, img_shape: tuple):
    if isinstance(pixel_coords, np.ndarray):
        img_shape = np.array(img_shape)
    elif isinstance(pixel_coords, torch.Tensor):
        img_shape = torch.tensor(img_shape).to(pixel_coords.device)
    else:
        raise TypeError("pixel_coords must be a numpy array or a torch tensor")

    normalized_coords = pixel_coords / (img_shape - 1) * 2 - 1
    return normalized_coords


def norm_to_voxel_coords(normalized_coords, img_shape: tuple):
    if isinstance(normalized_coords, np.ndarray):
        img_shape = np.array(img_shape)
    elif isinstance(normalized_coords, torch.Tensor):
        img_shape = torch.tensor(img_shape).to(normalized_coords.device)
    else:
        raise TypeError("normalized_coords must be a numpy array or a torch tensor")

    voxel_coords = (normalized_coords + 1) / 2 * (img_shape - 1)
    return voxel_coords


def compute_landmark_distance(
    predicted_landmarks: torch.Tensor,
    gt_landmarks: torch.Tensor,
    img_shape: tuple,
    img_spacing: tuple,
    snap_to_voxel: bool = True,
):
    pred_voxel_coords = norm_to_voxel_coords(predicted_landmarks, img_shape)
    gt_voxel_coords = norm_to_voxel_coords(gt_landmarks, img_shape)

    if snap_to_voxel:
        pred_voxel_coords = torch.round(pred_voxel_coords)
        gt_voxel_coords = torch.round(gt_voxel_coords)

    img_spacing = torch.tensor(img_spacing).to(predicted_landmarks.device)

    pred_mm = pred_voxel_coords * img_spacing
    gt_mm = gt_voxel_coords * img_spacing

    dists = torch.norm(pred_mm - gt_mm, dim=-1)

    return dists.mean(), dists.std()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims[dimension] = 1

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), flatten=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    if flatten:
        coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    return coordinate_tensor


def tensor_to_image(img, target_shape=(512, 512), normalize=False):
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
    img = img.reshape(target_shape)
    img = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


def displacement_slice_signed(U):
    """
    Parameters
    ----------
    U : np.ndarray
        Displacement field with shape (Nx, Ny, Nz, 3)

    Returns
    -------
    images : tuple of np.ndarray
        Three RGB arrays of shape (Nx, Nz, 3), dtype uint8
        with diverging color map (negative=blue, zero=white, positive=red).
    """
    U = U.detach().cpu().numpy()
    Nx, Nz, C = U.shape
    assert C == 3, "Last dimension must have size 3 (vector field)"

    # Extract components for the slice
    Vx = U[..., 0]
    Vy = U[..., 1]
    Vz = U[..., 2]

    # 1) GLOBAL NORMALIZATION FACTOR (shared by all components)
    max_abs = np.max(np.abs(U)) + 1e-12
    Vx_n = Vx / max_abs
    Vy_n = Vy / max_abs
    Vz_n = Vz / max_abs

    def signed_to_rgb(arr):
        """Map normalized array [-1,1] → RGB diverging (blue↔white↔red)."""

        H, W = arr.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        pos = arr >= 0
        neg = arr < 0

        # Positive → white → red
        rgb[pos, 0] = 255
        rgb[pos, 1] = (255 * (1 - arr[pos])).astype(np.uint8)
        rgb[pos, 2] = (255 * (1 - arr[pos])).astype(np.uint8)

        # Negative → white → blue
        rgb[neg, 2] = 255
        rgb[neg, 1] = (255 * (1 + arr[neg])).astype(np.uint8)
        rgb[neg, 0] = (255 * (1 + arr[neg])).astype(np.uint8)

        return rgb

    img_x = signed_to_rgb(Vx_n)
    img_y = signed_to_rgb(Vy_n)
    img_z = signed_to_rgb(Vz_n)

    return img_x, img_y, img_z


def read_image(
    filepath, normalize=False, clip=None
):  # new_spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    image = sitk.ReadImage(str(filepath))
    # image = resample_image_spacing(
    #    image, new_spacing=new_spacing, resampler=sitk.sitkNearestNeighbor
    # )
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, 2)
    if clip is not None:
        image = np.clip(image, clip[0], clip[1])
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())
    return image


def read_landmarks(filepath, delimiter=","):
    landmarks = np.loadtxt(filepath, delimiter=delimiter)
    return landmarks


def norm_point_in_voxel_mask(point, mask):
    point = norm_to_voxel_coords(point, mask.shape)
    point = torch.round(point).to(dtype=int)

    return mask[point[..., 0], point[..., 1], point[..., 2]]


def points_inside_mask(pts, mask):
    pts = np.asarray(pts).astype(int)

    # 1. Check bounds
    max_z, max_y, max_x = mask.shape
    in_bounds = (
        (0 <= pts[:, 0])
        & (pts[:, 0] < max_z)
        & (0 <= pts[:, 1])
        & (pts[:, 1] < max_y)
        & (0 <= pts[:, 2])
        & (pts[:, 2] < max_x)
    )

    if not np.all(in_bounds):
        print("Some points are outside the array entirely")
        return False

    # 2. Check mask values
    mask_values = mask[pts[:, 0], pts[:, 1], pts[:, 2]]
    return mask_values


def idir_compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def _path_constructor(loader, node):
    parts = loader.construct_sequence(node)
    return Path(*parts)


def load_config(path):
    path = Path(path)
    loader = yaml.FullLoader
    loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _path_constructor
    )
    loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath", _path_constructor
    )
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg


def save_config(path, cfg):
    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(serialize_config(cfg), f)


def crop_landmarks_to_bbox(landmarks, bbox):
    landmarks[..., 0] = landmarks[..., 0] - bbox[0].start
    landmarks[..., 1] = landmarks[..., 1] - bbox[1].start
    landmarks[..., 2] = landmarks[..., 2] - bbox[2].start

    return landmarks


def crop_to_mask(
    crop_mask,
    sampling_mask,
    image_fixed,
    image_moving,
    lung_mask_fixed,
    lung_mask_moving,
    total_segmentation_mask_fixed,
    total_segmentation_mask_moving,
    body_mask_fixed,
    fixed_landmarks,
    moving_landmarks,
    bbox=None,
):
    if bbox is None:
        bbox = get_bounding_box(torch.from_numpy(crop_mask), padding=2)

    fixed_landmarks[..., 0] = fixed_landmarks[..., 0] - bbox[0].start
    fixed_landmarks[..., 1] = fixed_landmarks[..., 1] - bbox[1].start
    fixed_landmarks[..., 2] = fixed_landmarks[..., 2] - bbox[2].start

    moving_landmarks[..., 0] = moving_landmarks[..., 0] - bbox[0].start
    moving_landmarks[..., 1] = moving_landmarks[..., 1] - bbox[1].start
    moving_landmarks[..., 2] = moving_landmarks[..., 2] - bbox[2].start

    return (
        bbox,
        sampling_mask[bbox],
        image_fixed[bbox],
        image_moving[bbox],
        lung_mask_fixed[bbox],
        lung_mask_moving[bbox],
        total_segmentation_mask_fixed[bbox]
        if total_segmentation_mask_fixed is not None
        else None,
        total_segmentation_mask_moving[bbox]
        if total_segmentation_mask_moving is not None
        else None,
        body_mask_fixed[bbox] if body_mask_fixed is not None else None,
        fixed_landmarks,
        moving_landmarks,
    )


def load_full_dirlab(case_folder, phases):
    images = []
    masks = []
    body_masks = []
    vessel_maps = []
    vessel_masks = []
    landmarks = {}
    extreme_landmarks = {}

    for phase in phases:
        mask = read_image(case_folder / f"segmentations_{phase:02d}.nii")

        body_mask = read_image(
            case_folder / f"segmentations_{phase:02d}" / "body.nii.gz"
        )

        vessel_mask = read_image(
            case_folder / "lung_vessels" / f"lung_vessels_{phase:02d}.nii"
        )

        vessel_map = np.load(
            case_folder / "lung_vessels" / f"lung_vessels_prob_{phase:02d}.npz"
        )["probabilities"][1]
        vessel_map = np.swapaxes(vessel_map, 0, 2)

        image = read_image(
            case_folder / "images" / f"phase_{phase:02d}.nii",
            normalize=True,
        )

        image_spacing = sitk.ReadImage(
            case_folder / "images" / f"phase_{phase:02d}.nii"
        ).GetSpacing()

        if phase <= 5:
            landmarks_path = case_folder / "landmarks_raw" / f"landmarks_{phase}.csv"
            if landmarks_path.exists():
                landmarks[phase] = read_landmarks(landmarks_path)
        if phase in [0, 5]:
            extreme_landmarks_path = (
                case_folder / "landmarks_raw" / f"extreme_landmarks_{phase}.csv"
            )
            if extreme_landmarks_path.exists():
                extreme_landmarks[phase] = read_landmarks(extreme_landmarks_path)

        images.append(image)
        masks.append(mask)
        body_masks.append(body_mask)
        vessel_masks.append(vessel_mask)
        vessel_maps.append(vessel_map)

    return {
        "images": images,
        "masks": masks,
        "body_masks": body_masks,
        "vessel_masks": vessel_masks,
        "vessel_maps": vessel_maps,
        "image_spacing": image_spacing,
        "landmarks": landmarks,
        "extreme_landmarks": extreme_landmarks,
    }


def load_and_crop_full_dirlab(
    case_folder: Path,
    phases: list[int],
    lung_classes: list[int] = [10, 11, 12, 13, 14],
    padding: int = 2,
    vessel_threshold: float = 0.5,
    crop: bool = True,
) -> dict:
    data = load_full_dirlab(case_folder, phases)

    images = data["images"]
    original_shape = images[0].shape
    masks = data["masks"]
    body_masks = data["body_masks"]
    vessel_masks = data["vessel_masks"]
    vessel_maps = data["vessel_maps"]
    landmarks = data["landmarks"]
    extreme_landmarks = data["extreme_landmarks"]

    lung_masks = [np.where(np.isin(mask, lung_classes), 1, 0) for mask in masks]
    lung_masks = np.stack(lung_masks, axis=0)

    union_lung_mask = np.where(np.sum(lung_masks, axis=0) > 0, 1, 0)
    if crop:
        bbox = get_bounding_box(torch.from_numpy(union_lung_mask), padding=padding)

        images = [image[bbox] for image in images]
        masks = [mask[bbox] for mask in masks]
        body_masks = [body_mask[bbox] for body_mask in body_masks]
        vessel_masks = [vessel_mask[bbox] for vessel_mask in vessel_masks]
        vessel_maps = [vessel_map[bbox] for vessel_map in vessel_maps]
        lung_masks = lung_masks[:, bbox[0], bbox[1], bbox[2]]
        union_lung_mask = union_lung_mask[bbox]

        landmarks = {
            phase: crop_landmarks_to_bbox(lm, bbox) for phase, lm in landmarks.items()
        }
        extreme_landmarks = {
            phase: crop_landmarks_to_bbox(lm, bbox)
            for phase, lm in extreme_landmarks.items()
        }
    else:
        bbox = None

    vessel_masks = [
        (vessel_map > vessel_threshold).astype(np.uint8) for vessel_map in vessel_maps
    ]

    return {
        "images": images,
        "original_shape": original_shape,
        "masks": masks,
        "body_masks": body_masks,
        "vessel_masks": vessel_masks,
        "vessel_maps": vessel_maps,
        "lung_masks": lung_masks,
        "union_lung_mask": union_lung_mask,
        "bbox": bbox,
        "landmarks": landmarks,
        "extreme_landmarks": extreme_landmarks,
        "image_spacing": data["image_spacing"],
    }


def load_dirlab(case_folder, fixed_phase, moving_phase, all_masks=True, clip=None):
    lung_mask_folder = case_folder / "masks"
    lung_mask_fixed = read_image(
        lung_mask_folder / f"lung_phase_{fixed_phase:02d}.nii.gz"
    )
    lung_mask_moving = read_image(
        lung_mask_folder / f"lung_phase_{moving_phase:02d}.nii.gz"
    )

    if all_masks:
        total_segmentation_mask_fixed = read_image(
            case_folder / f"segmentations_{fixed_phase:02d}.nii"
        )
        total_segmentation_mask_moving = read_image(
            case_folder / f"segmentations_{moving_phase:02d}.nii"
        )

        body_mask_fixed = read_image(
            case_folder / f"segmentations_{fixed_phase:02d}" / "body.nii.gz"
        )
    else:
        total_segmentation_mask_fixed = None
        total_segmentation_mask_moving = None
        body_mask_fixed = None

    image_fixed = read_image(
        case_folder / "images" / f"phase_{fixed_phase:02d}.nii",
        normalize=True,
        clip=clip,
    )
    image_moving = read_image(
        case_folder / "images" / f"phase_{moving_phase:02d}.nii",
        normalize=True,
        clip=clip,
    )

    image_spacing = sitk.ReadImage(
        case_folder / "images" / f"phase_{fixed_phase:02d}.nii"
    ).GetSpacing()

    # fixed_landmarks = read_landmarks(
    #    case_folder / "landmarks" / "fixed_landmarks_05_to_00.csv"
    # )
    # moving_landmarks = read_landmarks(
    #    case_folder / "landmarks" / "moving_landmarks_05_to_00.csv"
    # )
    fixed_landmarks = read_landmarks(
        case_folder / "landmarks_raw" / f"landmarks_{fixed_phase}.csv"
    )
    moving_landmarks = read_landmarks(
        case_folder / "landmarks_raw" / f"landmarks_{moving_phase}.csv"
    )

    return (
        image_fixed,
        image_moving,
        lung_mask_fixed,
        lung_mask_moving,
        total_segmentation_mask_fixed,
        total_segmentation_mask_moving,
        body_mask_fixed,
        fixed_landmarks,
        moving_landmarks,
        image_spacing,
    )


def gaussian_blur_3d(img, sigma):
    """
    img: (1, 1, Dx, Dy, Dz) float32 or float16
    sigma: float (0 = no blur)
    returns: same shape tensor
    """
    if sigma <= 0:
        return img

    # Kernel size = 6σ + 1 (odd)
    size = 2 * sigma + 1
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1

    size = int(size)

    # Create 1D gaussian kernel
    coords = torch.arange(size, device=img.device, dtype=img.dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()

    # Reshape into separable 3D kernels
    gX = g.view(1, 1, size, 1, 1)
    gY = g.view(1, 1, 1, size, 1)
    gZ = g.view(1, 1, 1, 1, size)

    # Apply 3 separable convolutions (VERY fast)
    out = F.conv3d(img, gX, padding=(size // 2, 0, 0), groups=1)
    out = F.conv3d(out, gY, padding=(0, size // 2, 0), groups=1)
    out = F.conv3d(out, gZ, padding=(0, 0, size // 2), groups=1)

    return out


def get_interpolation_coords(coords, image_shape):
    """
    coords: (N, 3) normalized in [-1, 1]
    returns: (N)
    """
    N, _ = coords.shape
    Dx, Dy, Dz = image_shape

    # -----------------------------
    # 1. Convert normalized coords to voxel indices
    # -----------------------------
    x = (coords[..., 0] + 1) * 0.5 * (Dx - 1)
    y = (coords[..., 1] + 1) * 0.5 * (Dy - 1)
    z = (coords[..., 2] + 1) * 0.5 * (Dz - 1)

    # Integer voxel corners
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Clamp
    x0 = torch.clamp(x0, 0, Dx - 1)
    y0 = torch.clamp(y0, 0, Dy - 1)
    z0 = torch.clamp(z0, 0, Dz - 1)
    x1 = torch.clamp(x1, 0, Dx - 1)
    y1 = torch.clamp(y1, 0, Dy - 1)
    z1 = torch.clamp(z1, 0, Dz - 1)

    # Distances (fractional part)
    xd = x - x0.float()
    yd = y - y0.float()
    zd = z - z0.float()

    return x0, x1, y0, y1, z0, z1, xd, yd, zd


def trilinear_interpolation(img, x0, x1, y0, y1, z0, z1, xd, yd, zd):
    c000 = img[x0, y0, z0]
    c100 = img[x1, y0, z0]
    c010 = img[x0, y1, z0]
    c001 = img[x0, y0, z1]
    c101 = img[x1, y0, z1]
    c011 = img[x0, y1, z1]
    c110 = img[x1, y1, z0]
    c111 = img[x1, y1, z1]

    # -----------------------------
    # 3. Trilinear interpolation
    # -----------------------------
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    out = c0 * (1 - zd) + c1 * zd

    return out  # shape (B, N)


def upsert_result(json_path, new_result):
    """
    new_result must contain: case, ref_phase, phase
    """
    json_path = Path(json_path)

    # Load existing results (or start fresh)
    if json_path.exists():
        with json_path.open("r") as f:
            results = json.load(f)
    else:
        results = []

    # Key fields
    key_fields = ("case", "ref_phase", "phase")

    # Try to find an existing entry
    for i, entry in enumerate(results):
        if all(entry[k] == new_result[k] for k in key_fields):
            results[i] = new_result  # overwrite entire entry
            break
    else:
        # Not found → append
        results.append(new_result)

    # Save back to file
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)


def serialize_config(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_config(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_config(v) for v in obj]
    else:
        return obj


def compute_tre_dense_dvf(
    moving_landmarks: np.ndarray,
    fixed_landmarks: np.ndarray,
    vector_field: np.ndarray | None = None,
    image_spacing=(1.0, 1.0, 1.0),
    snap_to_voxel: bool = False,
    axis: int | None = None,
) -> (np.ndarray | None, np.ndarray | None):
    if vector_field is not None:
        # order 1: linear interpolation if vector field at fixed landmarks
        displacement_x = ndi.map_coordinates(
            vector_field[0], fixed_landmarks.T, order=1
        )
        displacement_y = ndi.map_coordinates(
            vector_field[1], fixed_landmarks.T, order=1
        )
        displacement_z = ndi.map_coordinates(
            vector_field[2], fixed_landmarks.T, order=1
        )
        displacement = np.array((displacement_x, displacement_y, displacement_z)).T
        fixed_landmarks_warped = fixed_landmarks + displacement
    else:
        fixed_landmarks_warped = fixed_landmarks

    if snap_to_voxel:
        fixed_landmarks_warped = np.round(fixed_landmarks_warped)

    if axis is not None:
        axis_slicing = np.index_exp[:, axis : axis + 1]
        fixed_landmarks_warped = fixed_landmarks_warped[axis_slicing]
        moving_landmarks = moving_landmarks[axis_slicing]
        image_spacing = image_spacing[axis]

    tre = np.linalg.norm(
        (fixed_landmarks_warped - moving_landmarks) * image_spacing, axis=1
    )
    return tre, fixed_landmarks_warped
