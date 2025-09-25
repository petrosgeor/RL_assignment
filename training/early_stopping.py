from pathlib import Path
from training.utils import save_model


class EarlyStopping:
    """Tracks best success rate and signals when to stop training early."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = True,
        save_best: bool = False,
        best_path = None,
    ) -> None:
        # Validate patience and store configuration
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.verbose = verbose
        self.save_best = save_best
        # Resolve path for saving best model (optional)
        default_path = Path("saved_models/pro_model.best.zip")
        self.best_path = Path(best_path) if best_path else default_path
        self.best_path = self.best_path.resolve() if not self.best_path.is_absolute() else self.best_path

        # Internal tracking state
        self.best_score = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, success_rate: float, model = None) -> bool:
        """Updates early-stopping state and returns True when stopping."""

        score = float(success_rate)
        # Initialize best on first call and optionally save

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                 print(f"[early-stop] initial best success={score:.2%}")
            self._maybe_save(model)
            return self.early_stop

        # No improvement beyond min_delta: increment patience counter
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                 print(
                    f"[early-stop] no improvement ({score:.2%}); counter={self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                # Patience exhausted: signal early stop
                self.early_stop = True
                if self.verbose:
                     print(
                        f"[early-stop] stopping triggered (best={self.best_score:.2%})"
                    )
        else:
            # Improvement observed: reset counter and update best
            self.best_score = score
            self.counter = 0
            if self.verbose:
                 print(f"[early-stop] new best success={score:.2%}")
            self._maybe_save(model)

        return self.early_stop

    def best(self) -> float:
        """Returns the best success rate observed so far."""

        return self.best_score

    def _maybe_save(self, model) -> None:
        # Save current model to best_path if enabled and model provided
        if not self.save_best or model is None:
            return
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, self.best_path)
