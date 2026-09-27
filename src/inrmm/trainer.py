import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from inrmm.compat import PathLike
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from inrmm.deformation import jacobian_determinant

from inrmm.metrics import dice_score
from inrmm.utils import (
    compute_landmark_distance,
    get_interpolation_coords,
    pixel_coords_to_norm,
    make_coordinate_tensor,
)
from inrmm.base_trainer import BaseTrainer


class DualMotionModel(nn.Module):
    def __init__(self, forward_model: nn.Module, backward_model: nn.Module):
        super().__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model

    def forward(self, x, direction="forward"):
        if direction == "forward":
            return self.forward_model(x)
        if direction == "backward":
            return self.backward_model(x)
        raise ValueError(f"Unknown direction: {direction}")


class MotionModelTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        images: list[torch.Tensor],
        lung_vessel_maps: list[torch.Tensor],
        lung_masks: list[torch.Tensor],
        joint_lung_mask: torch.Tensor | None,
        signal: torch.Tensor,
        reference_phase: int,
        landmarks: torch.Tensor,
        image_spacing: tuple[float, float],
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        run_folder: PathLike,
        aim_repo: PathLike,
        loss_function: nn.Module,
        spatial_reg_function: nn.Module,
        temporal_reg_function: nn.Module,
        val_loader: DataLoader,
        experiment_name: str,
        device: str = "cuda",
        config: dict | None = None,
        track_every: int = 1,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            run_folder=run_folder,
            aim_repo=aim_repo,
            loss_function=loss_function,
            val_loader=val_loader,
            experiment_name=experiment_name,
            device=device,
            track_every=track_every,
        )

        if isinstance(model, DualMotionModel):
            self.forward_model = model.forward_model
            self.backward_model = model.backward_model
        else:
            raise RuntimeError("MotionModelTrainer requires DualMotionModel")

        self.images = torch.stack(images, dim=0).to(device)  # [P, D, H, W]

        self.lung_vessel_maps = torch.stack(lung_vessel_maps, dim=0).to(device)
        self.lung_masks = torch.stack(lung_masks, dim=0).to(device)  # [P, D, H, W]
        self.joint_lung_mask = (
            joint_lung_mask.to(device) if joint_lung_mask is not None else None
        )  # [D, H, W]
        self.reference_phase = reference_phase

        self.image_shape = images[0].shape  # D, H, W
        self.image_spacing = image_spacing

        self.signal = signal.to(device)  # P x k

        self.landmarks = {
            phase: pixel_coords_to_norm(
                torch.from_numpy(lm).float().to(device), self.image_shape
            )
            for phase, lm in landmarks.items()
        }

        self.spatial_reg_function = spatial_reg_function
        self.temporal_reg_function = temporal_reg_function
        self.scheduler = scheduler
        self.config = config
        self._cached_mask_indices = {}
        self._cached_grid = None
        self._cached_transformer = None

        self._training_images = self.images

    def trilinear_interpolation(
        self,
        phases,
        x0,
        x1,
        y0,
        y1,
        z0,
        z1,
        xd,
        yd,
        zd,
        lung_mask: bool = False,
        vessel_map: bool = False,
    ):
        if lung_mask and vessel_map:
            raise ValueError("Only one of lung_mask and vessel_map can be True.")

        if lung_mask:
            c000 = self.lung_masks[phases, x0, y0, z0]
            c100 = self.lung_masks[phases, x1, y0, z0]
            c010 = self.lung_masks[phases, x0, y1, z0]
            c001 = self.lung_masks[phases, x0, y0, z1]
            c101 = self.lung_masks[phases, x1, y0, z1]
            c011 = self.lung_masks[phases, x0, y1, z1]
            c110 = self.lung_masks[phases, x1, y1, z0]
            c111 = self.lung_masks[phases, x1, y1, z1]
        elif vessel_map:
            c000 = self.lung_vessel_maps[phases, x0, y0, z0]
            c100 = self.lung_vessel_maps[phases, x1, y0, z0]
            c010 = self.lung_vessel_maps[phases, x0, y1, z0]
            c001 = self.lung_vessel_maps[phases, x0, y0, z1]
            c101 = self.lung_vessel_maps[phases, x1, y0, z1]
            c011 = self.lung_vessel_maps[phases, x0, y1, z1]
            c110 = self.lung_vessel_maps[phases, x1, y1, z0]
            c111 = self.lung_vessel_maps[phases, x1, y1, z1]
        else:
            c000 = self._training_images[phases, x0, y0, z0]
            c100 = self._training_images[phases, x1, y0, z0]
            c010 = self._training_images[phases, x0, y1, z0]
            c001 = self._training_images[phases, x0, y0, z1]
            c101 = self._training_images[phases, x1, y0, z1]
            c011 = self._training_images[phases, x0, y1, z1]
            c110 = self._training_images[phases, x1, y1, z0]
            c111 = self._training_images[phases, x1, y1, z1]

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

    def _get_base_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._cached_grid is None
            or self._cached_grid.device != device
            or self._cached_grid.dtype != dtype
        ):
            d, h, w = self.image_shape
            z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
            y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
            x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            self._cached_grid = torch.stack([xx, yy, zz], dim=-1)
        return self._cached_grid

    def _warp_with_grid_sample(
        self, image: torch.Tensor, disp_norm: torch.Tensor
    ) -> torch.Tensor:
        # image: (1, 1, D, H, W); disp_norm: (D, H, W, 3) in axis order (D, H, W).
        # grid_sample expects last dim as (x, y, z) = (W, H, D), so remap:
        #   x <- disp[..., 2], y <- disp[..., 1], z <- disp[..., 0]
        base_grid = self._get_base_grid(image.device, image.dtype)

        disp_grid = torch.empty_like(disp_norm)
        disp_grid[..., 0] = disp_norm[..., 2]
        disp_grid[..., 1] = disp_norm[..., 1]
        disp_grid[..., 2] = disp_norm[..., 0]

        grid = (base_grid + disp_grid).unsqueeze(0)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _sample_dense_batch(self):
        cfg = self.config
        device = self.device
        phases = torch.arange(self.images.shape[0], dtype=torch.long, device=device)

        B = phases.numel()
        N = cfg["sampling"]["dense_points_per_batch"]

        if B == 0:
            raise RuntimeError("No phases available for dense sampling.")

        # default to uniform dense sampling in normalized coordinate space [-1, 1]
        dense_coords = torch.rand((B, N, 3), dtype=torch.float32, device=device) * 2 - 1

        if cfg["sampling"]["continuous"] and cfg["sampling"]["from_mask"]:
            # sample coords from lung mask voxels with jitter
            if "joint" not in self._cached_mask_indices:
                if self.joint_lung_mask is None:
                    raise RuntimeError("joint_lung_mask is required for mask sampling")
                self._cached_mask_indices["joint"] = torch.nonzero(
                    self.joint_lung_mask > 0, as_tuple=False
                )

            mask_indices = self._cached_mask_indices["joint"]
            if mask_indices.numel() > 0:
                idx = torch.randint(0, mask_indices.shape[0], (B, N), device=device)
                vox = mask_indices[idx].to(dtype=torch.float32)
                vox = vox + (torch.rand_like(vox) - 0.5)
                shape = torch.tensor(
                    self.image_shape, device=device, dtype=torch.float32
                ).view(1, 1, 3)
                dense_coords = (vox / (shape - 1.0)) * 2.0 - 1.0

        dense_phases = phases.view(B, 1, 1).repeat(1, N, 1)
        return dense_coords, dense_phases

    def train_on_batch(self, data):
        if not self.forward_model.training:
            self.forward_model.train()
        if not self.backward_model.training:
            self.backward_model.train()
        self.optimizer.zero_grad(None)
        dense_coords, dense_phases = self._sample_dense_batch()
        if (
            self.config["loss"]["jacobian_weight"] > 0
            or self.config["loss"]["laplacian_weight"] > 0
        ):
            dense_coords = dense_coords.requires_grad_(True)

        B, N, C = dense_coords.shape
        coords = dense_coords.view(B * N, C)

        phases = dense_phases.view(B * N)

        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
            surrogates = self.signal[phases]  # .to(self.device)
            if self.config["loss"]["temporal_weight"] > 0:
                surrogates = surrogates.requires_grad_(True)

            model_input = torch.cat([coords, surrogates], dim=-1)  # B*N x (3 + 2)

            total_displ_fwd = self.forward_model(model_input)
            total_displ_bwd = self.backward_model(model_input)

            new_coords_fwd = coords + total_displ_fwd
            new_coords_bwd = coords + total_displ_bwd

            input_voxels = get_interpolation_coords(coords, self.image_shape)
            new_voxels_fwd = get_interpolation_coords(new_coords_fwd, self.image_shape)
            new_voxels_bwd = get_interpolation_coords(new_coords_bwd, self.image_shape)

            # Forward model: reference fixed image
            fixed_fwd = self.trilinear_interpolation(
                self.reference_phase, *input_voxels
            )
            moving_fwd = self.trilinear_interpolation(phases, *new_voxels_fwd)
            mask_fwd = self.trilinear_interpolation(
                self.reference_phase, *input_voxels, lung_mask=True
            )
            data_loss_fwd = self.loss_function(
                moving_fwd.view(B, N),
                fixed_fwd.view(B, N),
                mask=(mask_fwd > 0.5).view(B, N),
            )

            # Backward model: reference moving image
            fixed_bwd = self.trilinear_interpolation(phases, *input_voxels)
            moving_bwd = self.trilinear_interpolation(
                self.reference_phase, *new_voxels_bwd
            )
            mask_bwd = self.trilinear_interpolation(
                phases, *input_voxels, lung_mask=True
            )
            data_loss_bwd = self.loss_function(
                moving_bwd.view(B, N),
                fixed_bwd.view(B, N),
                mask=(mask_bwd > 0.5).view(B, N),
            )

            data_loss = data_loss_fwd + data_loss_bwd

            if (
                self.config["loss"]["jacobian_weight"] > 0
                or self.config["loss"]["laplacian_weight"] > 0
            ):
                spatial_reg_loss = self.spatial_reg_function(
                    input_coords=coords, output=total_displ_fwd
                ) + self.spatial_reg_function(
                    input_coords=coords, output=total_displ_bwd
                )
            else:
                spatial_reg_loss = torch.tensor(0.0, device=self.device)

            if self.config["loss"]["temporal_weight"] > 0:
                temporal_reg_loss = self.temporal_reg_function(
                    surrogates, total_displ_fwd
                ) + self.temporal_reg_function(surrogates, total_displ_bwd)
            else:
                temporal_reg_loss = torch.tensor(0.0, device=self.device)

            consistency_weight = self.config["loss"].get("consistency_weight", 0.0)
            if consistency_weight > 0:
                cycle_input_fwd = torch.cat([new_coords_fwd, surrogates], dim=-1)
                cycle_disp_fwd = self.backward_model(cycle_input_fwd)
                cycle_coords_fwd = new_coords_fwd + cycle_disp_fwd

                cycle_input_bwd = torch.cat([new_coords_bwd, surrogates], dim=-1)
                cycle_disp_bwd = self.forward_model(cycle_input_bwd)
                cycle_coords_bwd = new_coords_bwd + cycle_disp_bwd

                # Weight cycle errors by normalized voxel spacing so larger-voxel axes
                # contribute more, independent of axis grid size.
                spacing_scale = torch.tensor(
                    self.image_spacing, device=coords.device, dtype=coords.dtype
                )
                spacing_scale = spacing_scale / spacing_scale.mean().clamp_min(1e-8)
                cycle_error_fwd_scaled = (coords - cycle_coords_fwd) * spacing_scale
                cycle_error_bwd_scaled = (coords - cycle_coords_bwd) * spacing_scale

                cycle_loss_fwd = (cycle_error_fwd_scaled**2).sum(dim=1).mean()
                cycle_loss_bwd = (cycle_error_bwd_scaled**2).sum(dim=1).mean()
                consistency_loss = 0.5 * (cycle_loss_fwd + cycle_loss_bwd)
                consistency_loss = consistency_loss * consistency_weight
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)

            loss = data_loss + spatial_reg_loss + temporal_reg_loss + consistency_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
            learning_rate = self.scheduler.get_last_lr()
        else:
            learning_rate = self.optimizer.param_groups[0]["lr"]

        output = {
            "train_loss": loss.item(),
            "data_loss": data_loss.item(),
            "data_loss_fwd": data_loss_fwd.item(),
            "data_loss_bwd": data_loss_bwd.item(),
            "reg_loss": spatial_reg_loss.item(),
            "temporal_reg_loss": temporal_reg_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "learning_rate": learning_rate,
        }
        return output

    def validate_on_batch(self, data):
        lm_results = self.evaluate_on_landmarks()
        landmark_distances_mean = lm_results["mean"]
        landmark_distances_mean_world = lm_results["mean_world"]

        mean_landmark_displacement = 0.0
        mean_landmark_displacement_world = 0.0
        if len(landmark_distances_mean) > 0:
            mean_landmark_displacement = sum(landmark_distances_mean.values()) / len(
                landmark_distances_mean
            )
            mean_landmark_displacement_world = sum(
                landmark_distances_mean_world.values()
            ) / len(landmark_distances_mean_world)

        landmark_distances_mean_world = {
            f"val_landmark_distance_mean_phase_{phase}": lm_dist
            for phase, lm_dist in landmark_distances_mean_world.items()
        }
        # landmark_distances_std = {
        #    f"val_landmark_distance_std_phase_{phase}": lm_dist
        #    for phase, lm_dist in landmark_distances_std.items()
        # }
        results = self.calculate_lung_vessel_dice(phases=None)
        dice = 0
        map_mae = 0
        mse = 0
        folding = 0
        for r in results:
            dice += r["dice_score"]
            map_mae += r["map_mae"]
            mse += r["mse"]
            folding += r["folding_percentage"]
        output = {
            **landmark_distances_mean_world,
            # **landmark_distances_std,
            "val_mean_landmark_displacement_world": mean_landmark_displacement_world,
            "val_mean_landmark_displacement": mean_landmark_displacement,
            "val_mean_dice_score": dice / len(results),
            "val_map_mae": map_mae / len(results),
            "val_mean_mse": mse / len(results),
            "val_mean_folding_percentage": folding / len(results),
        }

        return output

    def evaluate_on_landmarks(self):
        self.forward_model.eval()

        landmark_distances_mean = {}
        landmark_distances_std = {}

        landmark_distances_mean_world = {}
        landmark_distances_std_world = {}

        for phase, lm in self.landmarks.items():
            fixed_landmarks = self.landmarks[self.reference_phase]
            moving_landmarks = lm

            surrogate = self.signal[phase].to(self.device)
            model_input = torch.cat(
                [fixed_landmarks, surrogate.repeat(fixed_landmarks.shape[0], 1)],
                dim=-1,
            )
            with torch.no_grad():
                displacements = self.forward_model(model_input)
            pred_landmarks = fixed_landmarks + displacements
            landmark_distance_mean, landmark_distance_std = compute_landmark_distance(
                pred_landmarks.detach(),
                moving_landmarks.detach(),
                img_shape=self.image_shape,
                img_spacing=self.image_spacing,
            )
            landmark_distances_mean[phase] = landmark_distance_mean.item()
            landmark_distances_std[phase] = landmark_distance_std.item()

            landmark_distance_mean_world, landmark_distance_std_world = (
                compute_landmark_distance(
                    pred_landmarks.detach(),
                    moving_landmarks.detach(),
                    img_shape=self.image_shape,
                    img_spacing=self.image_spacing,
                    snap_to_voxel=False,
                )
            )
            landmark_distances_mean_world[phase] = landmark_distance_mean_world.item()
            landmark_distances_std_world[phase] = landmark_distance_std_world.item()

        return {
            "mean": landmark_distances_mean,
            "std": landmark_distances_std,
            "mean_world": landmark_distances_mean_world,
            "std_world": landmark_distances_std_world,
        }

    def calculate_lung_vessel_dice(self, phases=None):
        set_train_mode = False
        if self.backward_model.training:
            self.backward_model.eval()
            set_train_mode = True

        output = []
        if phases is None:
            phases = range(self.images.shape[0])

        for phase in phases:
            # reference is moving, phase is fixed
            grid = make_coordinate_tensor(dims=self.image_shape, flatten=True).to(
                self.device
            )
            surrogate = self.signal[phase]

            model_input = torch.cat(
                [grid, surrogate.repeat(grid.shape[0], 1)],
                dim=-1,
            )

            deformations = []
            for _chunk in torch.chunk(model_input, chunks=10, dim=0):
                with torch.no_grad():
                    input_chunk = _chunk.to(self.device)
                    deformation_chunk = self.backward_model(input_chunk)
                    deformations.append(deformation_chunk)

            deformations = torch.cat(deformations, dim=0)
            deformations = deformations.view(self.image_shape + (3,))

            moving_map = (
                self.lung_vessel_maps[self.reference_phase]
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )  # 1 x 1 x D x H x W
            warped_map = self._warp_with_grid_sample(
                moving_map.to(self.device),
                deformations,
            )

            fixed_map = (
                self.lung_vessel_maps[phase]
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            lung_mask = self.lung_masks[phase].unsqueeze(0).unsqueeze(0).to(self.device)
            lung_mask = (lung_mask > 0.5).float()

            warped_map = warped_map * lung_mask
            fixed_map = fixed_map * lung_mask

            map_mae = (
                torch.abs(warped_map - fixed_map) * lung_mask
            ).sum() / lung_mask.sum()

            warped_mask = (warped_map >= 0.5).float()
            fixed_mask = (fixed_map >= 0.5).float()

            dice_score_val = dice_score(warped_mask, fixed_mask)

            moving_image = (
                self.images[self.reference_phase]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            warped_image = self._warp_with_grid_sample(
                moving_image,
                deformations,
            )
            fixed_image = self.images[phase].unsqueeze(0).unsqueeze(0).to(self.device)

            mse = (
                (warped_image - fixed_image) ** 2 * (lung_mask > 0.5).float()
            ).sum() / (lung_mask > 0.5).float().sum()

            # Jacobian-based folding is defined for voxel-space displacement.
            deformations_voxel = torch.empty_like(deformations)
            for axis in range(3):
                deformations_voxel[..., axis] = (
                    deformations[..., axis] * (self.image_shape[axis] - 1) / 2
                )
            det_j = jacobian_determinant(
                deformations_voxel.permute(3, 0, 1, 2)
                .unsqueeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            det_j = (
                det_j.detach().cpu().numpy()
                if isinstance(det_j, torch.Tensor)
                else det_j
            )

            lung_mask_np = (
                (lung_mask.squeeze(0).squeeze(0) > 0.5).detach().cpu().numpy()
            )
            if min(lung_mask_np.shape) > 4:
                cropped_lung = lung_mask_np[2:-2, 2:-2, 2:-2]
            else:
                cropped_lung = lung_mask_np
            if det_j.shape != cropped_lung.shape:
                min_shape = tuple(
                    min(det_dim, mask_dim)
                    for det_dim, mask_dim in zip(det_j.shape, cropped_lung.shape)
                )
                det_j = det_j[tuple(slice(0, size) for size in min_shape)]
                cropped_lung = cropped_lung[tuple(slice(0, size) for size in min_shape)]

            det_j_lung = det_j[cropped_lung]
            if det_j_lung.size == 0:
                folding_percentage = 0.0
                detj_std = None
            else:
                num_folding = np.sum(det_j_lung <= 0)
                total_voxels = det_j_lung.size
                folding_percentage = float(num_folding / total_voxels)
                detj_std = float(np.std(det_j_lung))
            output.append(
                {
                    "phase": phase,
                    "dice_score": dice_score_val.item(),
                    "map_mae": map_mae.item(),
                    "mse": mse.item(),
                    "folding_percentage": folding_percentage,
                    "detj_std": detj_std,
                }
            )

        if set_train_mode:
            self.backward_model.train()
        return output
