from __future__ import annotations

import copy
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from inrmm.dir_config import DEFAULT_DIRINR_CONFIG
from inrmm.model import HybridRegistrationNetwork, Siren
from inrmm.ncc import ncc_per_batch
from inrmm.regularizers import DetJLaplacianLoss
from inrmm.scheduler import ConvergenceChecker
from inrmm.utils import (
    gaussian_blur_3d,
    get_interpolation_coords,
    make_coordinate_tensor,
    trilinear_interpolation,
)


class DirINR:
    def __init__(self, config: dict | None = None):
        self.config = copy.deepcopy(DEFAULT_DIRINR_CONFIG)
        if config is not None:
            self._update_config(self.config, config)
        self._cached_mask_indices = None

    def register(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_lung_mask: np.ndarray | None = None,
        moving_lung_mask: np.ndarray | None = None,
        image_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str = "cuda",
    ) -> dict[str, np.ndarray]:
        cfg = self.config
        self._cached_mask_indices = {}

        sampling_mask_joint = self._build_joint_sampling_mask(
            fixed_lung_mask=fixed_lung_mask,
            moving_lung_mask=moving_lung_mask,
            fixed_image=fixed_image,
        )

        image_fixed = fixed_image
        image_moving = moving_image

        data_loss_fn = ncc_per_batch
        spatial_reg_function = DetJLaplacianLoss(
            detj_weight=cfg["loss"]["jacobian_weight"],
            laplace_weight=cfg["loss"]["laplace_weight"],
        )
        inverse_consistency_weight = float(
            cfg["loss"].get("inverse_consistency_weight", 0.1)
        )

        model_forward, is_dual = self._build_model(cfg)
        model_backward, _ = self._build_model(cfg)
        optimizer = torch.optim.AdamW(
            chain(model_forward.parameters(), model_backward.parameters()),
            lr=float(cfg["optimizer"]["lr"]),
        )

        device_t = torch.device(device)
        model_forward = model_forward.to(device_t)
        model_backward = model_backward.to(device_t)
        fixed_image_t = torch.from_numpy(image_fixed).to(device_t, dtype=torch.float32)
        moving_image_t = torch.from_numpy(image_moving).to(
            device_t, dtype=torch.float32
        )
        sampling_mask_joint_t = torch.from_numpy(sampling_mask_joint).to(
            device_t, dtype=torch.bool
        )
        fixed_lung_mask_t = torch.from_numpy(
            np.ones_like(image_fixed, dtype=np.float32)
            if fixed_lung_mask is None
            else fixed_lung_mask.astype(np.float32)
        ).to(device_t)
        moving_lung_mask_t = torch.from_numpy(
            np.ones_like(image_moving, dtype=np.float32)
            if moving_lung_mask is None
            else moving_lung_mask.astype(np.float32)
        ).to(device_t)

        blur_sigmas = self._resolve_blur_sigmas(cfg)
        total_steps = int(cfg["training"]["total_steps"])
        convergence_checker = ConvergenceChecker(
            patience=int(cfg["training"].get("blur_convergence_patience", 50)),
            threshold=float(cfg["training"].get("blur_convergence_threshold", 0.005)),
            threshold_mode=str(
                cfg["training"].get("blur_convergence_threshold_mode", "rel")
            ),
        )
        scheduler_type = str(cfg["scheduler"].get("type", "cosine")).lower()
        reset_scheduler_on_blur_change = bool(
            cfg["scheduler"].get("reset_on_blur_change", True)
        )
        fine_start_stage = int(cfg["training"].get("fine_start_stage", 1))
        fine_start_stage = min(max(fine_start_stage, 0), len(blur_sigmas) - 1)

        i_stage = 0
        sigma = float(blur_sigmas[i_stage])
        fixed_stage, moving_stage = self._get_blurred_images(
            fixed_image=fixed_image_t,
            moving_image=moving_image_t,
            sigma=sigma,
        )
        scheduler = self._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=int(cfg["scheduler"]["warmup_steps"]),
            n_steps_stage=total_steps,
            end_lr=float(cfg["optimizer"]["end_lr"]),
        )

        progress = tqdm(
            range(total_steps),
            desc=(
                f"DirINR stage {i_stage + 1}/{len(blur_sigmas)} "
                f"(sigma={float(sigma):.2f})"
            ),
            unit="step",
        )
        for i_step in progress:
            if is_dual and i_stage >= fine_start_stage:
                if not model_forward.fine_branch_is_active:
                    model_forward.turn_on_fine_branch()
                if not model_backward.fine_branch_is_active:
                    model_backward.turn_on_fine_branch()

            optimizer.zero_grad(None)

            dense_coords = self._sample_dense_batch(
                sampling_mask=sampling_mask_joint_t,
                image_shape=fixed_image_t.shape,
                sampler=cfg["data"]["sampler"],
                n_points=cfg["data"]["dense_points_per_batch"],
                device=device_t,
                cache_key="joint",
            ).requires_grad_(True)

            coarse_fw, fine_fw, total_fw = self._predict_displacement(
                model=model_forward,
                is_dual=is_dual,
                coords=dense_coords,
            )
            coarse_bw, fine_bw, total_bw = self._predict_displacement(
                model=model_backward,
                is_dual=is_dual,
                coords=dense_coords,
            )

            new_coords_fw = dense_coords + total_fw
            new_coords_bw = dense_coords + total_bw

            input_voxels = get_interpolation_coords(dense_coords, fixed_image_t.shape)
            new_voxels_fw = get_interpolation_coords(new_coords_fw, fixed_image_t.shape)
            new_voxels_bw = get_interpolation_coords(
                new_coords_bw, moving_image_t.shape
            )

            fixed = trilinear_interpolation(fixed_stage, *input_voxels)
            moving_warped = trilinear_interpolation(moving_stage, *new_voxels_fw)
            moving = trilinear_interpolation(moving_stage, *input_voxels)
            fixed_warped = trilinear_interpolation(fixed_stage, *new_voxels_bw)
            mask_fwd = trilinear_interpolation(fixed_lung_mask_t, *input_voxels) > 0.5
            mask_bwd = trilinear_interpolation(moving_lung_mask_t, *input_voxels) > 0.5

            data_loss_fw = self._compute_masked_data_loss(
                data_loss_fn=data_loss_fn,
                moving=moving_warped,
                fixed=fixed,
                mask=mask_fwd,
            )
            data_loss_bw = self._compute_masked_data_loss(
                data_loss_fn=data_loss_fn,
                moving=fixed_warped,
                fixed=moving,
                mask=mask_bwd,
            )
            data_loss = 0.5 * (data_loss_fw + data_loss_bw)
            loss = data_loss

            reg_loss = torch.tensor(0.0, device=device_t)
            if cfg["loss"]["jacobian_weight"] > 0 or cfg["loss"]["laplace_weight"] > 0:
                reg_fw = spatial_reg_function(
                    input_coords=dense_coords,
                    output=coarse_fw,
                )
                reg_bw = spatial_reg_function(
                    input_coords=dense_coords,
                    output=coarse_bw,
                )
                if fine_fw is not None:
                    reg_fw = reg_fw + float(
                        cfg["loss"]["fine_reg_scale"]
                    ) * spatial_reg_function(
                        input_coords=dense_coords,
                        output=fine_fw,
                    )
                if fine_bw is not None:
                    reg_bw = reg_bw + float(
                        cfg["loss"]["fine_reg_scale"]
                    ) * spatial_reg_function(
                        input_coords=dense_coords,
                        output=fine_bw,
                    )
                reg_loss = 0.5 * (reg_fw + reg_bw)
                loss = loss + reg_loss

            inverse_consistency_loss = torch.tensor(0.0, device=device_t)
            if inverse_consistency_weight > 0:
                _, _, bw_at_fw = self._predict_displacement(
                    model=model_backward,
                    is_dual=is_dual,
                    coords=new_coords_fw,
                )
                _, _, fw_at_bw = self._predict_displacement(
                    model=model_forward,
                    is_dual=is_dual,
                    coords=new_coords_bw,
                )
                consistency_fw = (total_fw + bw_at_fw).pow(2).sum(dim=-1).mean()
                consistency_bw = (total_bw + fw_at_bw).pow(2).sum(dim=-1).mean()
                inverse_consistency_loss = 0.5 * (consistency_fw + consistency_bw)
                loss = loss + inverse_consistency_weight * inverse_consistency_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                if scheduler._step_count >= scheduler.t_total:
                    scheduler = None

            if (
                sigma > 0
                and i_stage < len(blur_sigmas) - 1
                and convergence_checker.update(float(loss.detach().item()))
            ):
                convergence_checker.reset()
                i_stage += 1
                sigma = float(blur_sigmas[i_stage])
                progress.set_description(
                    f"DirINR stage {i_stage + 1}/{len(blur_sigmas)} "
                    f"(sigma={float(sigma):.2f})"
                )
                fixed_stage, moving_stage = self._get_blurred_images(
                    fixed_image=fixed_image_t,
                    moving_image=moving_image_t,
                    sigma=sigma,
                )
                if reset_scheduler_on_blur_change:
                    remaining_steps = total_steps - (i_step + 1)
                    if remaining_steps <= 0:
                        scheduler = None
                        continue
                    scheduler = self._build_scheduler(
                        optimizer=optimizer,
                        scheduler_type=scheduler_type,
                        warmup_steps=int(cfg["scheduler"]["warmup_steps"]),
                        n_steps_stage=remaining_steps,
                        end_lr=float(cfg["optimizer"]["end_lr"]),
                    )

        dvf_forward = self._compute_dense_dvf(
            model=model_forward, image_shape=image_fixed.shape, device=device
        )
        dvf_backward = self._compute_dense_dvf(
            model=model_backward, image_shape=moving_image.shape, device=device
        )

        return {
            "forward_dvf": dvf_forward,
            "backward_dvf": dvf_backward,
        }

    def _compute_masked_data_loss(
        self,
        data_loss_fn,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if int(mask.sum().item()) >= 2:
            moving_masked = moving[mask].view(1, -1)
            fixed_masked = fixed[mask].view(1, -1)
            return data_loss_fn(moving_masked, fixed_masked)
        return data_loss_fn(moving.view(1, -1), fixed.view(1, -1))

    def _predict_displacement(
        self,
        model: torch.nn.Module,
        is_dual: bool,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if is_dual and model.fine_branch_is_active:
            coarse_displ, fine_displ = model(coords, train=True)
            total_displ = coarse_displ + fine_displ
            return coarse_displ, fine_displ, total_displ
        coarse_displ = model(coords, train=True)
        return coarse_displ, None, coarse_displ

    def _build_model(self, cfg: dict) -> tuple[torch.nn.Module, bool]:
        model_type = cfg["model"].get("type", "Siren")
        layers = cfg["model"]["layers"]
        model_type_normalized = str(model_type).lower()

        if model_type_normalized == "dual":
            if len(layers) < 3:
                raise ValueError(
                    "Dual model requires at least 3 entries in model.layers "
                    "(in, hidden, out)"
                )
            hidden_features = layers[1]
            if any(layer != hidden_features for layer in layers[1:-1]):
                raise ValueError(
                    "Dual model expects constant hidden width in model.layers "
                    f"but got {layers}"
                )
            model = HybridRegistrationNetwork(
                in_features=layers[0],
                out_features=layers[-1],
                hidden_features=hidden_features,
                hidden_layers=len(layers) - 3,
                coarse_omega0=cfg["model"]["coarse_omega0"],
                fine_omega0=cfg["model"]["fine_omega0"],
            )
            return model, True

        if model_type_normalized != "siren":
            raise ValueError("model.type must be 'Siren' or 'Dual'")
        model = Siren(layers=layers, omega=cfg["model"]["siren_freq"])
        return model, False

    def _resolve_blur_sigmas(self, cfg: dict) -> list[float]:
        training_cfg = cfg.get("training", {})
        if "blur_sigmas" in training_cfg:
            sigmas = [float(s) for s in training_cfg["blur_sigmas"]]
        elif "blur_sigma" in cfg:
            sigma = float(cfg["blur_sigma"])
            sigmas = [sigma, 0.0] if sigma > 0 else [0.0]
        else:
            sigmas = [0.0]

        if not sigmas:
            sigmas = [0.0]
        if any(s < 0 for s in sigmas):
            raise ValueError(f"All blur sigmas must be >= 0, but got {sigmas}")
        if sigmas[-1] != 0.0:
            sigmas.append(0.0)
        return sigmas

    def _split_steps(self, total_steps: int, n_stages: int) -> list[int]:
        if n_stages <= 0:
            raise ValueError("n_stages must be > 0")
        base_steps = total_steps // n_stages
        remainder = total_steps % n_stages
        steps = [base_steps for _ in range(n_stages)]
        for i in range(remainder):
            steps[i] += 1
        return steps

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        warmup_steps: int,
        n_steps_stage: int,
        end_lr: float,
    ) -> WarmupCosineSchedule | None:
        if scheduler_type in {"none", ""}:
            return None
        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return self._build_stage_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            n_steps_stage=n_steps_stage,
            end_lr=end_lr,
        )

    def _get_blurred_images(
        self,
        fixed_image: torch.Tensor,
        moving_image: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma <= 0:
            return fixed_image, moving_image

        fixed_blurred = gaussian_blur_3d(fixed_image[None, None], sigma)[0, 0]
        moving_blurred = gaussian_blur_3d(moving_image[None, None], sigma)[0, 0]
        return fixed_blurred, moving_blurred

    def _build_stage_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        n_steps_stage: int,
        end_lr: float,
    ) -> WarmupCosineSchedule:
        if n_steps_stage <= 0:
            raise ValueError("n_steps_stage must be > 0 to build a scheduler")
        warmup_steps_stage = min(max(warmup_steps, 0), n_steps_stage)
        return WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps_stage,
            t_total=n_steps_stage,
            end_lr=end_lr,
        )

    def _build_sampling_mask(
        self,
        fixed_lung_mask: np.ndarray | None,
        moving_lung_mask: np.ndarray | None,
        fixed_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if fixed_lung_mask is None:
            sampling_mask = np.ones_like(fixed_image, dtype=bool)
            crop_mask = sampling_mask
            return sampling_mask, crop_mask

        if moving_lung_mask is None:
            moving_lung_mask = fixed_lung_mask

        sampling_mask = fixed_lung_mask.astype(bool)
        crop_mask = np.logical_or(fixed_lung_mask, moving_lung_mask)
        return sampling_mask, crop_mask

    def _build_joint_sampling_mask(
        self,
        fixed_lung_mask: np.ndarray | None,
        moving_lung_mask: np.ndarray | None,
        fixed_image: np.ndarray,
    ) -> np.ndarray:
        if fixed_lung_mask is None and moving_lung_mask is None:
            return np.ones_like(fixed_image, dtype=bool)
        if fixed_lung_mask is None:
            return moving_lung_mask.astype(bool)
        if moving_lung_mask is None:
            return fixed_lung_mask.astype(bool)
        return np.logical_or(fixed_lung_mask > 0, moving_lung_mask > 0)

    def _compute_dense_dvf(
        self, model: torch.nn.Module, image_shape: tuple[int, int, int], device: str
    ) -> np.ndarray:
        model.eval()
        grid = make_coordinate_tensor(dims=image_shape, flatten=True).to(device)
        deformations = []
        for _chunk in torch.chunk(grid, chunks=10, dim=0):
            with torch.no_grad():
                deformations.append(model(_chunk))
        deformations = torch.cat(deformations, dim=0)
        deformations = deformations.view(image_shape + (3,))
        return deformations.detach().cpu().numpy()

    def _sample_dense_batch(
        self,
        sampling_mask: torch.Tensor,
        image_shape: tuple[int, int, int],
        sampler: str,
        n_points: int,
        device: torch.device,
        cache_key: str,
    ) -> torch.Tensor:
        if self._cached_mask_indices is None:
            self._cached_mask_indices = {}
        if cache_key not in self._cached_mask_indices:
            self._cached_mask_indices[cache_key] = torch.nonzero(
                sampling_mask > 0, as_tuple=False
            )
        mask_indices = self._cached_mask_indices[cache_key]

        if mask_indices.numel() == 0:
            coords = torch.rand((n_points, 3), device=device) * 2 - 1
        else:
            idx = torch.randint(0, mask_indices.shape[0], (n_points,), device=device)
            vox = mask_indices[idx].float()
            if sampler == "continuous":
                vox = vox + (torch.rand_like(vox) - 0.5)
            dx, dy, dz = image_shape
            coords = torch.empty_like(vox)
            coords[:, 0] = (vox[:, 0] / (dx - 1)) * 2 - 1
            coords[:, 1] = (vox[:, 1] / (dy - 1)) * 2 - 1
            coords[:, 2] = (vox[:, 2] / (dz - 1)) * 2 - 1

        return coords

    def _update_config(self, base: dict, update: dict) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
