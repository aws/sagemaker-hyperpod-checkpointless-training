"""Environment Variable Validation."""

# Standard Library
import os
from dataclasses import dataclass
from typing import Any, Callable

# First Party
from hyperpod_checkpointless_training.inprocess.exception import (
    HyperPodCheckpointlessValidationError,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnvVarSpec:
    """
    Environment Variable specification.

    Hard validation should raise errors.
    Soft validation should log warnings.
    """

    name: str
    type: Callable = str
    required: bool = False
    min: float | None = None
    max: float | None = None
    choices: list | None = None
    soft_min: float | None = None
    soft_max: float | None = None
    soft_choices: list | None = None

    def _validate_hard_bounds(self, typed_value: int | float):
        """
        Validate hard bounds.

        Raises errors.
        """
        if self.min is not None and typed_value < self.min:
            raise HyperPodCheckpointlessValidationError(
                f"{self.name}={typed_value} < min {self.min}"
            )
        if self.max is not None and typed_value > self.max:
            raise HyperPodCheckpointlessValidationError(
                f"{self.name}={typed_value} > max {self.max}"
            )

    def _validate_soft_bounds(self, typed_value: int | float):
        """
        Validate soft bounds.

        Logs warnings.
        """
        if self.soft_min is not None and typed_value < self.soft_min:
            logger.warning(
                f"{self.name}={typed_value} below recommended min: {self.soft_min}"
            )
        if self.soft_max is not None and typed_value > self.soft_max:
            logger.warning(
                f"{self.name}={typed_value} above recommended max: {self.soft_max}"
            )

    def _validate_choices(self, typed_value: Any):
        """
        Validate hard choices.

        Raises errors.
        """
        if self.choices and typed_value not in self.choices:
            raise HyperPodCheckpointlessValidationError(
                f"{self.name}={typed_value} not in allowed {self.choices}"
            )

    def _validate_soft_choices(self, typed_value: Any):
        """
        Validate soft choices.

        Logs warnings.
        """
        if self.soft_choices and typed_value not in self.soft_choices:
            logger.warning(
                f"{self.name}={typed_value} not in recommended {self.soft_choices}"
            )

    def validate(self, value: str | None) -> Any:
        """
        Validate a single variable.

        Raises errors for hard validation checks, and warnings for soft validation checks.
        """
        # Handle missing values
        if value is None or value == "":
            if self.required:
                raise HyperPodCheckpointlessValidationError(
                    f"Missing required environment variable: {self.name}"
                )
            return

        # Validate type conversion
        try:
            typed_value = self.type(value)
        except ValueError:
            raise HyperPodCheckpointlessValidationError(
                f"{self.name}: cannot convert '{value}' to {self.type.__name__}"
            )

        # Assume if hard or soft bounds are set, then type is int or float.
        self._validate_hard_bounds(typed_value)
        self._validate_soft_bounds(typed_value)

        self._validate_choices(typed_value)
        self._validate_soft_choices(typed_value)


@dataclass
class NProcPNodeEnvVarSpec(EnvVarSpec):
    """N_PROC_PER_NODE Custom Environment Variable specification."""

    @staticmethod
    def get_num_gpus() -> int:
        """Return number of visible GPUs using torch."""
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")

        return 0

    def __post_init__(self):
        """Post init."""
        # Validate against # gpus in node as warning maximum
        if self.soft_max is None:
            self.soft_max = NProcPNodeEnvVarSpec.get_num_gpus()

    def _validate_soft_bounds(self, typed_value: int | float):
        """Validate soft bounds."""
        if self.soft_min is not None and typed_value < self.soft_min:
            logger.warning(
                f"{self.name}={typed_value} below recommended min={self.soft_min}"
            )
        if self.soft_max is not None and typed_value > self.soft_max:
            logger.warning(
                f"{self.name}={typed_value} above recommended max of # gpus per node={self.soft_max}"
            )


class EnvManager:
    """Manages environment variable validation."""

    def __init__(self, specs: list[EnvVarSpec] | None = None):
        """Initialize environment manager."""
        PORT_MIN = 1024
        PORT_MAX = 65535

        if specs:
            self.specs = specs
        else:
            self.specs = [
                # Rootless
                NProcPNodeEnvVarSpec("N_PROC_PER_NODE", type=int, soft_min=1),
                EnvVarSpec(
                    "HPCT_PORT_BASE", type=int, soft_min=PORT_MIN, soft_max=PORT_MAX
                ),
                EnvVarSpec("NCCL_SOCKET_RETRY_DELAY", type=int, soft_min=0),
                # TCPStore Removal
                EnvVarSpec("HPCT_USE_ROOTLESS", type=int, soft_choices=[0, 1]),
                EnvVarSpec("TORCH_DIST_INIT_BARRIER", type=int, soft_choices=[0, 1]),
                EnvVarSpec(
                    "TORCH_NCCL_GET_PORT_BASE",
                    type=int,
                    soft_min=PORT_MIN,
                    soft_max=PORT_MAX,
                ),
                EnvVarSpec(
                    "TORCH_GLOO_GET_PORT_BASE",
                    type=int,
                    soft_min=PORT_MIN,
                    soft_max=PORT_MAX,
                ),
            ]

    def validate(self):
        """
        Validate all environment variables.
        """
        errors = []
        for spec in self.specs:
            value = os.getenv(spec.name, None)
            if value is None:
                logger.info(f"Validating {spec.name} (unset)")
            else:
                logger.info(f"Validating {spec.name}='{value}'")
            try:
                spec.validate(value)
            except HyperPodCheckpointlessValidationError as e:
                errors.append(str(e))

        if errors:
            formatted_errors = ", ".join(errors)
            raise HyperPodCheckpointlessValidationError(f"({formatted_errors})")

        logger.info("Environment variable validation passed!")


if __name__ == "__main__":
    env_manager = EnvManager()
    env_manager.validate()
