class ConvergenceChecker:
    """
    Lightweight convergence checker following PyTorch's ReduceLROnPlateau pattern.
    Much faster than statistical approaches - just tracks best loss and patience counter.
    """

    def __init__(self, patience=50, threshold=1e-3, threshold_mode="rel"):
        """
        Args:
            patience: Number of steps without improvement before declaring convergence
            threshold: Minimum improvement to be considered "better"
            threshold_mode: 'rel' for relative improvement, 'abs' for absolute
        """
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_stable_steps = 0

    def update(self, loss: float) -> bool:
        """
        Update with new loss value.

        Returns:
            True if converged (stable for 'patience' steps), False otherwise
        """
        if self.best is None:
            self.best = loss
            return False

        # Check if current loss is significantly better than best
        if self.threshold_mode == "rel":
            # Relative improvement: loss must be < best * (1 - threshold)
            # e.g., threshold=0.001 means loss must improve by 0.1%
            threshold_value = self.best * (1.0 - self.threshold)
            is_better = loss < threshold_value
        else:  # 'abs'
            # Absolute improvement: loss must be < best - threshold
            is_better = loss < (self.best - self.threshold)

        if is_better:
            self.best = loss
            self.num_stable_steps = 0
        else:
            self.num_stable_steps += 1

        # Converged if no significant improvement for 'patience' steps
        return self.num_stable_steps >= self.patience

    def reset(self):
        """Reset checker (e.g., after transitioning blur levels)"""
        self.best = None
        self.num_stable_steps = 0

    @property
    def is_tracking(self) -> bool:
        """Whether checker has started tracking (received at least one loss)"""
        return self.best is not None
