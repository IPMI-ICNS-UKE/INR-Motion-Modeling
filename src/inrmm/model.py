import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    def __init__(
        self, layers, weight_init=True, omega=30, temporal_omega=None, final_scaling=0.1
    ):
        super().__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.final_scaling = final_scaling
        if temporal_omega is not None:
            self.register_buffer(
                "omega1",
                torch.tensor([omega, omega, omega, temporal_omega, temporal_omega]),
            )
        else:
            self.omega1 = None

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, train=False):
        """The forward function of the network."""

        for i, layer in enumerate(self.layers[:-1]):
            if i == 0 and self.omega1 is not None:
                x = torch.sin(layer(x * self.omega1))
            else:
                x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.final_scaling * self.layers[-1](x)


class LinearSiren(nn.Module):
    """
    Linear respiratory model:
    delta(x, A, v) = M(x) @ [A, v], where M(x) in R^{3x2}.
    The model accepts 5D input (x, A, v) to keep the same interface,
    but only uses spatial coords to predict M(x).
    """

    def __init__(
        self,
        layers,
        weight_init=True,
        omega=30,
    ):
        super().__init__()
        self.siren = Siren(layers=layers, weight_init=weight_init, omega=omega)

    def forward(self, x, train=False):
        coords = x[:, :3]
        surrogates = x[:, 3:5]
        coeffs = self.siren(coords)
        coeffs = coeffs.view(-1, 3, 2)
        delta = torch.bmm(coeffs, surrogates.unsqueeze(-1)).squeeze(-1)
        return delta


class SineLayer(nn.Module):
    """SIREN layer with sine activation."""

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer uses 1/in_features
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers use SIREN gain scaled by omega_0
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


def init_siren_output_layer(layer, hidden_features, omega):
    """Final linear output layer initialization for SIREN."""
    with torch.no_grad():
        bound = np.sqrt(6 / hidden_features) / omega
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.zero_()


class HybridRegistrationNetwork(nn.Module):
    """
    Hybrid SIREN network for multi-scale INR-based deformable registration.

    - Coarse branch: low-frequency SIREN (global deformation)
    - Fine branch: high-frequency SIREN (local residual deformation)
    - Output = coarse + fine
    """

    def __init__(
        self,
        in_features=3,
        out_features=3,
        hidden_features=256,
        hidden_layers=2,
        coarse_omega0=10,
        fine_omega0=30,
        coarse_scale=0.5,
        fine_scale=0.1,
        fine_branch_is_active=False,
    ):
        super().__init__()

        coarse_layers = []
        coarse_layers.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=coarse_omega0
            )
        )
        for _ in range(hidden_layers - 1):
            coarse_layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=coarse_omega0,
                )
            )

        coarse_out = nn.Linear(hidden_features, out_features)
        init_siren_output_layer(coarse_out, hidden_features, omega=coarse_omega0)
        coarse_layers.append(coarse_out)

        self.coarse_branch = nn.Sequential(*coarse_layers)

        fine_layers = []
        fine_layers.append(
            SineLayer(in_features, hidden_features, is_first=True, omega_0=fine_omega0)
        )
        for _ in range(hidden_layers - 1):
            fine_layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=fine_omega0,
                )
            )
        fine_out = nn.Linear(hidden_features, out_features)
        init_siren_output_layer(fine_out, hidden_features, omega=fine_omega0)
        fine_layers.append(fine_out)

        self.fine_branch = nn.Sequential(*fine_layers)

        self.coarse_scale = nn.Parameter(torch.tensor(coarse_scale))
        self.fine_scale = nn.Parameter(torch.tensor(fine_scale))
        self.fine_branch_is_active = fine_branch_is_active
        self.coarse_is_frozen = False

    def turn_on_fine_branch(self):
        self.fine_branch_is_active = True
        self.fine_scale = nn.Parameter(self.coarse_scale.detach() * 0.1)

    def freeze_coarse_branch(self):
        for param in self.coarse_branch.parameters():
            param.requires_grad = False
        self.coarse_is_frozen = True

    def forward(self, coords, train=False):
        coarse_disp = self.coarse_branch(coords)

        if not self.fine_branch_is_active:
            return self.coarse_scale * coarse_disp

        fine_disp = self.fine_branch(coords)
        if train:
            return self.coarse_scale * coarse_disp, self.fine_scale * fine_disp
        return self.coarse_scale * coarse_disp + self.fine_scale * fine_disp

    def get_coarse_displacement(self, coords):
        return self.coarse_scale * self.coarse_branch(coords)

    def get_fine_displacement(self, coords):
        return self.fine_scale * self.fine_branch(coords)
