"""Unit tests for env_validation.py."""

# Standard Library
import os
import unittest
from unittest.mock import patch

# Third Party
import pytest

# First Party
from hyperpod_checkpointless_training.inprocess.env_validation import (
    EnvManager,
    EnvVarSpec,
    HyperPodCheckpointlessValidationError,
    NProcPNodeEnvVarSpec,
)


class TestEnvVarSpec(unittest.TestCase):
    """Test cases for EnvVarSpec class."""

    def test_validate_string_value(self):
        """Test validating a string value."""
        spec = EnvVarSpec(name="TEST_VAR", type=str)
        spec.validate("test_value")

    def test_validate_int_value(self):
        """Test validating an integer value."""
        spec = EnvVarSpec(name="TEST_INT", type=int)
        spec.validate("42")

    def test_validate_float_value(self):
        """Test validating a float value."""
        spec = EnvVarSpec(name="TEST_FLOAT", type=float)
        spec.validate("3.14")

    def test_validate_missing_required_value(self):
        """Test validating missing required value raises error."""
        spec = EnvVarSpec(name="REQUIRED_VAR", required=True)
        with pytest.raises(
            HyperPodCheckpointlessValidationError,
            match="Missing required environment variable: REQUIRED_VAR",
        ):
            spec.validate(None)

    def test_validate_empty_required_value(self):
        """Test validating empty required value raises error."""
        spec = EnvVarSpec(name="REQUIRED_VAR", required=True)
        with pytest.raises(
            HyperPodCheckpointlessValidationError,
            match="Missing required environment variable: REQUIRED_VAR",
        ):
            spec.validate("")

    def test_validate_missing_optional_value(self):
        """Test validating missing optional value returns None."""
        spec = EnvVarSpec(name="OPTIONAL_VAR", required=False)
        spec.validate(None)

    def test_validate_invalid_type_conversion(self):
        """Test validating with invalid type conversion raises error."""
        spec = EnvVarSpec(name="INT_VAR", type=int)
        with pytest.raises(
            HyperPodCheckpointlessValidationError,
            match="INT_VAR: cannot convert 'not_a_number' to int",
        ):
            spec.validate("not_a_number")

    def test_validate_value_below_min(self):
        """Test validating value below min raises error."""
        spec = EnvVarSpec(name="MIN_VAR", type=int, min=10)
        with pytest.raises(HyperPodCheckpointlessValidationError, match="< min 10"):
            spec.validate("5")

    def test_validate_value_above_max(self):
        """Test validating value above max raises error."""
        spec = EnvVarSpec(name="MAX_VAR", type=int, max=100)
        with pytest.raises(HyperPodCheckpointlessValidationError, match="> max 100"):
            spec.validate("150")

    def test_validate_value_within_bounds(self):
        """Test validating value within bounds succeeds."""
        spec = EnvVarSpec(name="BOUNDED_VAR", type=int, min=10, max=100)
        spec.validate("50")

    def test_validate_value_at_min_boundary(self):
        """Test validating value at min boundary succeeds."""
        spec = EnvVarSpec(name="BOUNDED_VAR", type=int, min=10)
        spec.validate("10")

    def test_validate_value_at_max_boundary(self):
        """Test validating value at max boundary succeeds."""
        spec = EnvVarSpec(name="BOUNDED_VAR", type=int, max=100)
        spec.validate("100")

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_value_below_soft_min(self, mock_logger):
        """Test validating value below soft min logs warning."""
        spec = EnvVarSpec(name="WARN_VAR", type=int, soft_min=50)
        spec.validate("30")
        mock_logger.warning.assert_called_once()
        assert "below recommended min" in mock_logger.warning.call_args[0][0]

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_value_above_soft_max(self, mock_logger):
        """Test validating value above soft max logs warning."""
        spec = EnvVarSpec(name="WARN_VAR", type=int, soft_max=100)
        spec.validate("150")
        mock_logger.warning.assert_called_once()
        assert "above recommended max" in mock_logger.warning.call_args[0][0]

    def test_validate_value_not_in_choices(self):
        """Test validating value not in choices raises error."""
        spec = EnvVarSpec(name="CHOICE_VAR", type=int, choices=[0, 1])
        with pytest.raises(
            HyperPodCheckpointlessValidationError, match="not in allowed"
        ):
            spec.validate("2")

    def test_validate_value_in_choices(self):
        """Test validating value in choices succeeds."""
        spec = EnvVarSpec(name="CHOICE_VAR", type=int, choices=[0, 1, 2])
        spec.validate("1")

    def test_validate_string_choices(self):
        """Test validating string value with choices."""
        spec = EnvVarSpec(
            name="CHOICE_STR", type=str, choices=["debug", "info", "error"]
        )
        spec.validate("info")

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_value_not_in_soft_choices(self, mock_logger):
        """Test validating value not in choices raises error."""
        spec = EnvVarSpec(name="CHOICE_VAR", type=int, soft_choices=[0, 1])
        spec.validate("2")
        mock_logger.warning.assert_called_once()

    def test_validate_float_with_bounds(self):
        """Test validating float value with bounds."""
        spec = EnvVarSpec(name="FLOAT_VAR", type=float, min=0.0, max=1.0)
        spec.validate("0.5")


class TestNProcPNodeEnvVarSpec(unittest.TestCase):
    """Test cases for NProcPNodeEnvVarSpec."""

    @patch("torch.cuda.device_count")
    def test_get_num_gpus_with_torch(self, mock_torch_device_count):
        """Test get_num_gpus with PyTorch available."""
        mock_torch_device_count.return_value = 4
        result = NProcPNodeEnvVarSpec.get_num_gpus()
        assert result == 4

    def test_get_num_gpus_without_torch(self):
        """Test get_num_gpus without PyTorch available."""
        with patch.dict("sys.modules", {"torch": None}):
            result = NProcPNodeEnvVarSpec.get_num_gpus()
            assert result == 0

    @patch("torch.cuda.device_count")
    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_get_num_gpus_torch_exception(self, mock_logger, mock_torch_device_count):
        """Test get_num_gpus when PyTorch raises exception."""
        mock_torch_device_count.side_effect = RuntimeError("CUDA not available")
        result = NProcPNodeEnvVarSpec.get_num_gpus()
        assert result == 0
        mock_logger.debug.assert_called_once()

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_value_below_soft_min(self, mock_logger):
        """Test validating value below soft min logs warning."""
        spec = NProcPNodeEnvVarSpec(name="N_PROC_PER_NODE", type=int, soft_min=1)
        spec.validate("0")
        mock_logger.warning.assert_called_once()
        assert "below recommended min" in mock_logger.warning.call_args[0][0]

    @patch("torch.cuda.device_count")
    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_value_above_soft_max(self, mock_logger, mock_torch_device_count):
        """Test validating value above soft max logs warning."""
        mock_torch_device_count.return_value = 4
        spec = NProcPNodeEnvVarSpec(name="N_PROC_PER_NODE", type=int, soft_min=1)
        spec.validate("5")
        mock_logger.warning.assert_called_once()
        assert "above recommended max" in mock_logger.warning.call_args[0][0]


class TestEnvManager(unittest.TestCase):
    """Test cases for EnvManager class."""

    def test_env_manager_initialization(self):
        """Test EnvManager initialization."""
        manager = EnvManager()
        assert len(manager.specs) > 0
        assert all(isinstance(spec, EnvVarSpec) for spec in manager.specs)

    def test_env_manager_has_required_specs(self):
        """Test EnvManager has all required environment variable specs."""
        manager = EnvManager()
        spec_names = [spec.name for spec in manager.specs]

        assert "N_PROC_PER_NODE" in spec_names
        assert "HPCT_PORT_BASE" in spec_names
        assert "NCCL_SOCKET_RETRY_DELAY" in spec_names
        assert "HPCT_USE_ROOTLESS" in spec_names
        assert "TORCH_DIST_INIT_BARRIER" in spec_names
        assert "TORCH_NCCL_GET_PORT_BASE" in spec_names
        assert "TORCH_GLOO_GET_PORT_BASE" in spec_names

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_success(self, mock_logger):
        """Test validate with all valid environment variables."""
        manager = EnvManager()

        with patch.dict(
            os.environ,
            {
                "N_PROC_PER_NODE": "4",
                "HPCT_PORT_BASE": "8080",
                "NCCL_SOCKET_RETRY_DELAY": "10",
                "HPCT_USE_ROOTLESS": "1",
                "TORCH_DIST_INIT_BARRIER": "0",
                "TORCH_NCCL_GET_PORT_BASE": "9000",
                "TORCH_GLOO_GET_PORT_BASE": "9100",
            },
        ):
            manager.validate()

        # No error should be raised
        # Should log validation passed
        assert any(
            "passed" in str(call).lower() for call in mock_logger.info.call_args_list
        )

    def test_validate_missing_required_variable(self):
        """Test validate with missing required variable."""
        manager = EnvManager(specs=[EnvVarSpec("REQUIRED_VAR", required=True)])

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                HyperPodCheckpointlessValidationError,
                match="Missing required environment variable: REQUIRED_VAR",
            ):
                manager.validate()

    def test_validate_invalid_type(self):
        """Test validate with invalid type."""
        manager = EnvManager()

        with patch.dict(
            os.environ, {"N_PROC_PER_NODE": "not_a_number", "HPCT_USE_ROOTLESS": "1"}
        ):
            with pytest.raises(
                HyperPodCheckpointlessValidationError, match="cannot convert"
            ):
                manager.validate()

    def test_validate_out_of_bounds(self):
        """Test validate with out of bounds value."""
        manager = EnvManager(specs=[EnvVarSpec("BOUNDED_VAR", type=int, min=1)])

        with patch.dict(
            os.environ,
            {"BOUNDED_VAR": "0", "HPCT_USE_ROOTLESS": "1"},
        ):
            with pytest.raises(HyperPodCheckpointlessValidationError, match="< min"):
                manager.validate()

    def test_validate_invalid_choice(self):
        """Test validate with invalid choice value."""
        manager = EnvManager(specs=[EnvVarSpec("CHOICE_VAR", type=int, choices=[0, 1])])

        with patch.dict(
            os.environ,
            {"CHOICE_VAR": "4"},
        ):
            with pytest.raises(
                HyperPodCheckpointlessValidationError, match="not in allowed"
            ):
                manager.validate()

    def test_validate_port_below_min(self):
        """Test validate with port below min."""
        manager = EnvManager(
            specs=[EnvVarSpec("PORT_VAR", type=int, min=1024, max=65535)]
        )

        with patch.dict(
            os.environ,
            {
                "PORT_VAR": "500",
            },
        ):
            with pytest.raises(
                HyperPodCheckpointlessValidationError, match="< min 1024"
            ):
                manager.validate()

    def test_validate_port_above_max(self):
        """Test validate with port above max."""
        manager = EnvManager(
            specs=[EnvVarSpec("PORT_VAR", type=int, min=1024, max=65535)]
        )

        with patch.dict(
            os.environ,
            {
                "PORT_VAR": "70000",
            },
        ):
            with pytest.raises(
                HyperPodCheckpointlessValidationError, match="> max 65535"
            ):
                manager.validate()

    def test_validate_multiple_errors(self):
        """Test validate with multiple errors."""
        manager = EnvManager(
            specs=[
                EnvVarSpec("MIN_VAR", type=int, min=1),
                EnvVarSpec("CHOICE_VAR", type=int, choices=[0, 1]),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "MIN_VAR": "0",
                "CHOICE_VAR": "5",
            },
        ):
            with pytest.raises(HyperPodCheckpointlessValidationError) as exc_info:
                manager.validate()

            # Should contain both errors
            error_msg = str(exc_info.value)
            assert "MIN_VAR" in error_msg and "CHOICE_VAR" in error_msg

    @patch("hyperpod_checkpointless_training.inprocess.env_validation.logger")
    def test_validate_optional_variables_missing(self, mock_logger):
        """Test validate with optional variables missing."""
        manager = EnvManager()

        with patch.dict(
            os.environ,
            {
                # Optional variables not set
            },
        ):
            manager.validate()

        # Should still pass validation
        assert any(
            "validation passed" in str(call).lower()
            for call in mock_logger.info.call_args_list
        )
