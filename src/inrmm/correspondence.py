from __future__ import annotations

import logging
import pickle
from functools import cached_property
from hashlib import sha256
from pathlib import Path

import numpy as np
from inrmm.compat import PathLike

logger = logging.getLogger(__name__)


class CorrespondenceModel:
    def __init__(self):
        self.coefficients: np.ndarray | None = None
        self.timesteps: int | None = None
        self.mean_signal: np.ndarray | None = None
        self.signal_n_dims: int | None = None
        self.mean_vector_field: np.ndarray | None = None
        self.spatial_shape = None
        self.signals: np.ndarray | None = None
        self.reference_phase: int | None = None

    @cached_property
    def model_hash(self) -> str:
        """Calculate a SHA256 hash of the correspondence model to uniquely
        identify it."""
        if not self.is_fitted:
            raise RuntimeError("Correspondence model is not fitted")

        hasher = sha256()
        hasher.update(self.coefficients.tobytes())
        hasher.update(self.timesteps.to_bytes(1, "big"))
        hasher.update(self.mean_signal.tobytes())
        hasher.update(self.mean_vector_field.tobytes())
        hasher.update(self.signals.tobytes())
        hasher.update(self.reference_phase.to_bytes(1, "big"))

        return hasher.hexdigest()

    def save(self, filepath: PathLike, include_model_hash: bool = True):
        filepath = Path(filepath)
        filepath = filepath.with_suffix(".pkl")
        if include_model_hash:
            filepath = filepath.with_name(
                f"{filepath.stem}_{self.model_hash[:7]}{filepath.suffix}"
            )
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "coefficients": self.coefficients,
                    "timesteps": self.timesteps,
                    "mean_signal": self.mean_signal,
                    "signal_n_dims": self.signal_n_dims,
                    "mean_vector_field": self.mean_vector_field,
                    "spatial_shape": self.spatial_shape,
                    "signals": self.signals,
                    "reference_phase": self.reference_phase,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: PathLike) -> CorrespondenceModel:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        correspondence_model = cls()
        for key, value in data.items():
            setattr(correspondence_model, key, value)

        return correspondence_model

    @property
    def is_fitted(self) -> bool:
        return all(
            v is not None
            for v in (self.coefficients, self.mean_signal, self.mean_vector_field)
        )

    def _regularize_matrix(
        self,
        matrix: np.ndarray,
        condition_number_treshold: float = 30.0,
        step_size: float = 1e-3,
    ) -> np.ndarray:
        """Regularize a matrix by adding a small value to the diagonal.

        This is called Tikhonov regularization and is used to avoid non-
        invertibility of the matrix. We regularize the matrix until its
        condition number is below a given threshold. If the condition
        number is already below the threshold, the matrix is not
        regularized.
        """

        tikhonov_regularization = 0.0
        regularizing_matrix = np.eye(matrix.shape[0]) * tikhonov_regularization

        # check if matrix is full rank => can be inverted without regularization
        if np.linalg.matrix_rank(matrix) == min(matrix.shape):
            logger.debug("Matrix is full rank")
            condition_number = np.linalg.cond(matrix)
        else:
            condition_number = float("inf")

        if condition_number > condition_number_treshold:
            logger.info("Iterative Tikhnov regularization started")
            while condition_number > condition_number_treshold:
                # add small value to diagonal
                tikhonov_regularization += step_size
                regularizing_matrix = np.eye(matrix.shape[0]) * tikhonov_regularization
                condition_number = np.linalg.cond(matrix + regularizing_matrix)
                logger.debug(
                    f"Regularized matrix with {tikhonov_regularization=} "
                    f"results in {condition_number=}"
                )
                if tikhonov_regularization > 1.0:
                    raise RuntimeError(
                        "Abort matrix regularization. Tikhnov regularization reached 1.0."
                    )

                if condition_number < condition_number_treshold:
                    logger.info("Iterative Tikhnov regularization finished")
                    break
            logger.info(f"Regularized matrix with {tikhonov_regularization=}")
        else:
            logger.debug(
                f"No regularization needed as "
                f"{condition_number=} <= {condition_number_treshold=}"
            )
        return matrix + regularizing_matrix

    def fit(
        self,
        vector_fields: np.ndarray,
        signals: np.ndarray,
        reference_phase: int = 2,
        centering: str = "ref",
    ):
        """Fit the correspondence model according to Wilms et al. (2014) using
        multivariate regression solved by ordinary least squares.

        References:
        [1] https://doi.org/10.1088/0031-9155/59/5/1147
        """
        # final shapes of used matrices:
        # vector_fields: (3*x*y*z, timesteps)
        # signals: (signal_n_dims, timesteps)
        # coefficients: (3*x*y*z, signal_n_dims)

        # input vector_fields is of shape (timesteps, 3, x, y, z)
        self.spatial_shape = vector_fields.shape[2:]
        self.timesteps = vector_fields.shape[0]
        # reshape vector_fields to matrix of shape (3*x*y*z, t)
        vector_fields = vector_fields.reshape(self.timesteps, -1).T
        # calculate mean along timesteps
        signals = signals.reshape(self.timesteps, -1).T
        self.signal_n_dims = signals.shape[0]
        # calculate mean along timesteps
        if centering == "ref":
            logger.info("Centering using reference phase")
            self.mean_vector_field = np.zeros_like(vector_fields[:, :1])
            self.mean_signal = np.zeros_like(signals[:, :1])
            vector_fields[:, reference_phase : reference_phase + 1] = 0.0
            assert signals[:, reference_phase : reference_phase + 1].min() == 0.0
            assert signals[:, reference_phase : reference_phase + 1].max() == 0.0

        else:
            self.mean_vector_field = np.mean(vector_fields, axis=1, keepdims=True)
            self.mean_signal = np.mean(signals, axis=1, keepdims=True)

        centered_vector_fields = vector_fields - self.mean_vector_field
        centered_signals = signals - self.mean_signal

        # calculating Moore-Penrose pseudo-inverse of signals matrix
        # Here: avoid non-invertibility by using Tikhonov regularization if needed

        if self.timesteps >= self.signal_n_dims:
            # time steps >= signal dimensions, i.e., n_rows >= n_columns
            #
            logger.info("time steps >= signal dimensions")
            covariance_matrix = centered_signals @ centered_signals.T
            covariance_matrix = self._regularize_matrix(covariance_matrix)
            centered_signals_pinv = centered_signals.T @ np.linalg.inv(
                covariance_matrix
            )
        else:
            logger.info("time steps < signal dimensions")
            covariance_matrix = centered_signals.T @ centered_signals
            covariance_matrix = self._regularize_matrix(covariance_matrix)
            centered_signals_pinv = (
                np.linalg.inv(covariance_matrix) @ centered_signals.T
            )

        self.coefficients = centered_vector_fields @ centered_signals_pinv
        self.signals = signals
        self.reference_phase = reference_phase

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Correspondence model is not fitted")
        if signal.shape != (self.signal_n_dims,):
            raise ValueError(
                f"Given signal has wrong shape. "
                f"Expected ({self.signal_n_dims},), but got {signal.shape}"
            )

        # input signal is of shape (signal_n_dims,)
        # reshape to (signal_n_dims, timestamps=1)
        signal = signal[:, None]
        centered_signal = signal - self.mean_signal

        # prediction is of shape (3*x*y*z, 1)
        prediction = self.mean_vector_field + self.coefficients @ centered_signal

        # reshape to (3, x, y, z)
        prediction = prediction.reshape(3, *self.spatial_shape)

        return prediction

    @classmethod
    def build_default(
        cls,
        images: np.ndarray,
        signals: np.ndarray,
        vector_fields: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        centering: str = "ref",
        device: str = "cuda",
        reference_phase: int = 2,
        phase_to_leave_out: int | None = None,
        masked_registration: bool = True,
        direction: str = "forward",
        return_vector_fields: bool = False,
    ):
        """Build a default correspondence model using the images, masks and
        signal."""
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        if vector_fields is None:
            raise ValueError(
                "Precomputed vector_fields are required. Registration backends are "
                "kept separate from the linear correspondence model."
            )

        vector_fields_fit = vector_fields
        signals_fit = signals
        reference_phase_fit = reference_phase
        if phase_to_leave_out is not None:
            n_phases = vector_fields.shape[0]
            if phase_to_leave_out < 0 or phase_to_leave_out >= n_phases:
                raise ValueError(
                    "phase_to_leave_out is out of bounds for given data: "
                    f"{phase_to_leave_out} not in [0, {n_phases - 1}]"
                )
            if n_phases <= 2:
                raise ValueError(
                    "Cannot leave one phase out with <= 2 phases. "
                    f"Got {n_phases} phases."
                )
            if centering == "ref" and phase_to_leave_out == reference_phase:
                raise ValueError(
                    "phase_to_leave_out cannot be the reference phase when centering='ref'."
                )

            keep_mask = np.ones(n_phases, dtype=bool)
            keep_mask[phase_to_leave_out] = False
            keep_indices = np.flatnonzero(keep_mask)
            vector_fields_fit = vector_fields[keep_indices]
            signals_fit = signals[keep_indices]

            # Fit uses indices within the filtered arrays.
            if phase_to_leave_out < reference_phase:
                reference_phase_fit = reference_phase - 1

        correspondence_model = cls()
        correspondence_model.fit(
            vector_fields=vector_fields_fit,
            signals=signals_fit,
            reference_phase=reference_phase_fit,
            centering=centering,
        )
        # Keep original phase indexing for metadata and downstream evaluation.
        correspondence_model.reference_phase = reference_phase

        if return_vector_fields:
            return correspondence_model, vector_fields

        return correspondence_model
