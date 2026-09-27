import torch
import torch.nn.functional as F


def idir_compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy."""

    jacobian_matrix = idir_compute_jacobian_matrix(
        input_coords, output, add_identity=False
    )

    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dy_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dz_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        dx_xyz[:, i, :] = idir_gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xyz[:, i, :] = idir_gradient(input_coords, jacobian_matrix[:, i, 1])
        dz_xyz[:, i, :] = idir_gradient(input_coords, jacobian_matrix[:, i, 2])

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    dz_xyz = torch.square(dz_xyz)

    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss += (
        2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + torch.mean(dy_xyz[:, :, 2])
    )

    return loss


def idir_compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = idir_gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


def idir_gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad


def laplacian_autograd(coords, y):
    """
    coords: (N,3) with requires_grad=True
    y:      (N,3) output of MLP at coords
    returns lap: (N,3) containing Δy1, Δy2, Δy3
    """

    N = coords.shape[0]
    lap = []

    for j in range(3):  # loop over output channels (small, cheap)
        # First-order gradient ∇ y_j
        grad_yj = torch.autograd.grad(
            y[:, j],
            coords,
            grad_outputs=torch.ones(N, device=y.device, dtype=y.dtype),
            create_graph=True,
            retain_graph=True,
        )[0]  # (N,3)

        # Compute divergence of grad_yj = Laplacian
        dxx = torch.autograd.grad(
            grad_yj[:, 0],
            coords,
            grad_outputs=torch.ones(N, device=y.device, dtype=y.dtype),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]

        dyy = torch.autograd.grad(
            grad_yj[:, 1],
            coords,
            grad_outputs=torch.ones(N, device=y.device, dtype=y.dtype),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1]

        dzz = torch.autograd.grad(
            grad_yj[:, 2],
            coords,
            grad_outputs=torch.ones(N, device=y.device, dtype=y.dtype),
            create_graph=True,
            retain_graph=True,
        )[0][:, 2]

        lap.append(dxx + dyy + dzz)

    return torch.stack(lap, dim=1)  # (N,3)


def bending_energy_laplacian(input_coords, output):
    lap = laplacian_autograd(input_coords, output)
    return (lap**2).mean()


def compute_jacobian(input_coords, output, add_identity=False):
    """
    Computes J = ∂f/∂x
    input_coords: (N,3) with requires_grad=True
    output: (N,3)

    returns J: (N,3,3)
    """
    N = input_coords.shape[0]
    J = []

    for j in range(3):
        grad_j = torch.autograd.grad(
            outputs=output[:, j],
            inputs=input_coords,
            grad_outputs=torch.ones(N, device=output.device),
            create_graph=True,
            retain_graph=True,
        )[0]  # (N,3)
        J.append(grad_j)

    J = torch.stack(J, dim=1)  # (N,3,3)

    if add_identity:
        J = J + torch.eye(3, device=J.device).unsqueeze(0)

    return J


def compute_laplacian_from_jacobian(J, input_coords):
    """
    Computes Δf_i(x) for each dimension using 2nd derivatives:
        Δf_i = ∂²f_i/∂x² + ∂²f_i/∂y² + ∂²f_i/∂z²

    J: (N,3,3) already in graph
    returns Laplacian: (N,3)
    """
    N, T = input_coords.shape
    device = J.device

    lap = torch.zeros(N, 3, device=device)

    # For each output channel f_i:
    for i in range(3):
        # Row i contains ∂f_i/∂x_j
        grad_fi = J[:, i, :]  # (N,3)

        # Compute div(grad_fi)
        for j in range(T):
            second_grad = torch.autograd.grad(
                outputs=grad_fi[:, j],
                inputs=input_coords,
                grad_outputs=torch.ones(N, device=device),
                create_graph=True,
                retain_graph=True,
            )[0]  # (N,3)

            lap[:, i] += second_grad[:, j]

    return lap  # (N,3)


def bending_energy_loss(input_coords, output):
    """
    Bending energy = ∫ ||Δ f(x)||^2 dx
    """
    J = compute_jacobian(input_coords, output)

    lap = compute_laplacian_from_jacobian(J, input_coords)

    # Norm squared of Laplacian vector
    bending = (lap.pow(2).sum(dim=1)).mean()
    return bending


def jacobian_det_loss(input_coords, output):
    """
    det(J) deviation penalty
    """
    J = compute_jacobian(input_coords, output, add_identity=False)
    detJ = torch.det(J)
    return (detJ - 1).abs().mean()


def jacobian_det_folding_loss(input_coords, output, t=None):
    """
    det(J) folding penalty
    """
    J = compute_jacobian(input_coords, output, add_identity=True)
    detJ = torch.det(J)

    return F.relu(-detJ).mean()


def VCC(input_coords, output, t=0.2):
    """
    det(J) deviation penalty
    """
    J = compute_jacobian(input_coords, output, add_identity=True)
    detJ = torch.det(J)
    mask = detJ >= t
    detJ_transformed = torch.where(
        mask,
        (detJ - 1) ** 2 / detJ,  # detJ >= t
        (1 - 1 / t**2) * detJ + 2 * (1 - t) / t,  # detJ < t
    )
    return detJ_transformed.mean()


def combined_loss(input_coords, output, t=0.2):
    """
    Computes both losses but reuses the Jacobian computation.
    """
    detJ = compute_jacobian(input_coords, output, add_identity=False)

    # --- Jacobian determinant loss ---
    mask = detJ >= t
    detJ_transformed = torch.where(
        mask,
        (detJ - 1) ** 2 / detJ,  # detJ >= t
        (1 - 1 / t**2) * detJ + 2 * (1 - t) / t,  # detJ < t
    )
    det_loss = detJ_transformed.mean()
    # det_loss = (torch.det(J) - 1).abs().mean()

    # --- Bending energy Laplacian loss ---
    lap = compute_laplacian_from_jacobian(detJ, input_coords)
    bending_loss = (lap.pow(2).sum(dim=1)).mean()

    return bending_loss + det_loss


def compute_jacobian_and_laplacian(coords, output, add_identity=False):
    """
    Efficiently compute both Jacobian determinant and Laplacian in one pass.

    Args:
        coords: (N, 3) with requires_grad=True
        output: (N, 3) displacement field
        add_identity: If True, compute det(I + J) instead of det(J)

    Returns:
        det_J: (N,) Jacobian determinants
        laplacian: (N, 3) Laplacian for each output component [Δu, Δv, Δw]
    """
    N = output.shape[0]
    device = output.device
    dtype = output.dtype

    # Initialize Jacobian matrix and Laplacian
    jacobian = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    laplacian = torch.zeros(N, 3, device=device, dtype=dtype)

    # Compute first and second derivatives
    for j in range(3):  # For each output dimension (u, v, w)
        # First derivatives: ∇output_j = [∂output_j/∂x, ∂output_j/∂y, ∂output_j/∂z]
        grad_j = torch.autograd.grad(
            output[:, j],
            coords,
            grad_outputs=torch.ones(N, device=device, dtype=dtype),
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 3)

        # Store in Jacobian
        jacobian[:, j, :] = grad_j

        # Second derivatives for Laplacian: Δoutput_j = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
        for k in range(3):  # For each spatial dimension
            second_deriv = torch.autograd.grad(
                grad_j[:, k],
                coords,
                grad_outputs=torch.ones(N, device=device, dtype=dtype),
                create_graph=True,
                retain_graph=True,
            )[0][:, k]  # Only diagonal: ∂²output_j/∂x_k²

            laplacian[:, j] += second_deriv

    # Add identity if computing deformation gradient (F = I + J)
    if add_identity:
        jacobian[:, 0, 0] += 1.0
        jacobian[:, 1, 1] += 1.0
        jacobian[:, 2, 2] += 1.0

    # Compute determinant using the rule of Sarrus (for 3x3)
    det_J = (
        jacobian[:, 0, 0]
        * (
            jacobian[:, 1, 1] * jacobian[:, 2, 2]
            - jacobian[:, 1, 2] * jacobian[:, 2, 1]
        )
        - jacobian[:, 0, 1]
        * (
            jacobian[:, 1, 0] * jacobian[:, 2, 2]
            - jacobian[:, 1, 2] * jacobian[:, 2, 0]
        )
        + jacobian[:, 0, 2]
        * (
            jacobian[:, 1, 0] * jacobian[:, 2, 1]
            - jacobian[:, 1, 1] * jacobian[:, 2, 0]
        )
    )

    return det_J, laplacian


def combined_jacobian_laplacian_loss_midl(input_coords, output, t=0.1):
    J = compute_jacobian(input_coords, output, add_identity=True)

    detJ = torch.det(J)
    jac_loss = F.relu(-detJ).mean()
    laplacian = compute_laplacian_from_jacobian(J, input_coords)
    lap_loss = (laplacian**2).mean()

    return jac_loss + lap_loss


class VCCLaplacianLoss(torch.nn.Module):
    def __init__(self, vcc_weight, laplace_weight, t=0.1):
        super().__init__()
        self.vcc_weight = vcc_weight
        self.laplace_weight = laplace_weight
        self.t = t

    def forward(self, input_coords, output):
        vcc_loss, lap_loss = self.vcc_laplacian_loss(input_coords, output)
        return self.vcc_weight * vcc_loss + self.laplace_weight * lap_loss

    def vcc_laplacian_loss(self, input_coords, output):
        detJ, laplacian = compute_jacobian_and_laplacian(
            input_coords, output, add_identity=True
        )

        # Jacobian folding loss
        mask = detJ >= self.t
        detJ_transformed = torch.where(
            mask,
            (detJ - 1) ** 2 / detJ,
            (1 - 1 / self.t**2) * detJ + 2 * (1 - self.t) / self.t,
        )
        jac_loss = detJ_transformed.mean()

        # Laplacian regularization
        lap_loss = (laplacian**2).mean()

        return jac_loss, lap_loss


class DetJLaplacianLoss(torch.nn.Module):
    def __init__(self, detj_weight, laplace_weight):
        super().__init__()
        self.detj_weight = detj_weight
        self.laplace_weight = laplace_weight

    def forward(self, input_coords, output):
        detJ, laplacian = compute_jacobian_and_laplacian(
            input_coords, output, add_identity=True
        )
        detj_loss = ((detJ - 1) ** 2).mean()
        lap_loss = (laplacian**2).mean()
        return self.detj_weight * detj_loss + self.laplace_weight * lap_loss


class DetJLoss(torch.nn.Module):
    def __init__(self, detj_weight):
        super().__init__()
        self.detj_weight = detj_weight

    def forward(self, input_coords, output):
        detJ = compute_jacobian(input_coords, output, add_identity=True)
        detj_loss = ((detJ - 1) ** 2).mean()
        return self.detj_weight * detj_loss


class TemporalLoss(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, temp, output):
        if self.weight == 0:
            return torch.tensor(0.0, device=output.device)

        tv_loss = 0

        for i in range(3):
            grad_temp = torch.autograd.grad(
                outputs=output[:, i],
                inputs=temp,
                grad_outputs=torch.ones_like(output[:, i]),
                create_graph=True,
                retain_graph=True,
            )[0]  # (N,2)

            tv_loss += torch.mean(grad_temp.pow(2).sum(dim=1))

        return self.weight * tv_loss


class TemporalCurvatureLoss(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input_coords, output):
        """
        Temporal curvature: sum of second derivatives w.r.t. temporal coords.
        Expects input_coords to be (N, 2): [amplitude, velocity].
        """
        if input_coords.shape[1] != 2:
            raise ValueError(
                f"TemporalCurvatureLoss expects input_coords with 2 dims, got {input_coords.shape}"
            )

        N = output.shape[0]
        lap = torch.zeros(N, 3, device=output.device, dtype=output.dtype)

        for i in range(3):
            # first derivatives wrt temporal coords
            grad_fi = torch.autograd.grad(
                outputs=output[:, i],
                inputs=input_coords,
                grad_outputs=torch.ones(N, device=output.device, dtype=output.dtype),
                create_graph=True,
                retain_graph=True,
            )[0]  # (N,2)

            # second derivatives sum over temporal dims
            for j in range(2):
                second_grad = torch.autograd.grad(
                    outputs=grad_fi[:, j],
                    inputs=input_coords,
                    grad_outputs=torch.ones(
                        N, device=output.device, dtype=output.dtype
                    ),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, j]
                lap[:, i] += second_grad

        curvature = (lap.pow(2).sum(dim=1)).mean()
        return self.weight * curvature


class MaskLoss(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, pred_classes, true_classes):
        if self.weight == 0:
            return torch.tensor(0.0, device=pred_classes.device)

        # masks are of shape (C, N) where C is number of classes
        loss = F.mse_loss(pred_classes, true_classes)
        return self.weight * loss
