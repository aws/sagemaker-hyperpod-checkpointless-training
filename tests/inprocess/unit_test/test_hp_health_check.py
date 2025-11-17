import datetime
import logging
import threading
import unittest
from unittest.mock import MagicMock, call, patch

from hyperpod_checkpointless_training.inprocess.exception import TimeoutError
from hyperpod_checkpointless_training.inprocess.health_check import (
    CudaHealthCheck,
    FaultCounter,
    FaultCounterExceeded,
    HealthCheck,
)
from hyperpod_checkpointless_training.inprocess.utils import HPState


class TestHealthCheckAbstractBase(unittest.TestCase):
    """Test the abstract base HealthCheck class"""

    def test_health_check_is_abstract(self):
        """Test that HealthCheck cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            HealthCheck()

    def test_health_check_abstract_method(self):
        """Test that __call__ is an abstract method"""

        class IncompleteHealthCheck(HealthCheck):
            pass

        with self.assertRaises(TypeError):
            IncompleteHealthCheck()

    def test_health_check_concrete_implementation(self):
        """Test that a concrete implementation can be created"""

        class ConcreteHealthCheck(HealthCheck):
            def __call__(self, state, train_ex=None):
                return state, train_ex

        health_check = ConcreteHealthCheck()
        state = HPState()
        result_state, result_ex = health_check(state)

        self.assertEqual(result_state, state)
        self.assertIsNone(result_ex)

    def test_health_check_concrete_implementation_with_exception(self):
        """Test concrete implementation with exception"""

        class ConcreteHealthCheck(HealthCheck):
            def __call__(self, state, train_ex=None):
                return state, train_ex

        health_check = ConcreteHealthCheck()
        state = HPState()
        exception = ValueError("test error")
        result_state, result_ex = health_check(state, exception)

        self.assertEqual(result_state, state)
        self.assertEqual(result_ex, exception)


class TestCudaHealthCheck(unittest.TestCase):
    """Test the CudaHealthCheck class"""

    def setUp(self):
        self.state = HPState()
        self.default_timeout = datetime.timedelta(seconds=30)

    def test_init_with_default_timeout(self):
        """Test CudaHealthCheck initialization with default timeout"""
        health_check = CudaHealthCheck()
        self.assertEqual(health_check.timeout, self.default_timeout)

    def test_init_with_custom_timeout(self):
        """Test CudaHealthCheck initialization with custom timeout"""
        custom_timeout = datetime.timedelta(seconds=60)
        health_check = CudaHealthCheck(custom_timeout)
        self.assertEqual(health_check.timeout, custom_timeout)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    def test_call_cuda_not_available(self, mock_is_initialized, mock_is_available):
        """Test call when CUDA is not available"""
        mock_is_available.return_value = False
        mock_is_initialized.return_value = False

        health_check = CudaHealthCheck()
        result_state, result_ex = health_check(self.state)

        self.assertEqual(result_state, self.state)
        self.assertIsNone(result_ex)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    def test_call_cuda_not_initialized(self, mock_is_initialized, mock_is_available):
        """Test call when CUDA is available but not initialized"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = False

        health_check = CudaHealthCheck()
        result_state, result_ex = health_check(self.state)

        self.assertEqual(result_state, self.state)
        self.assertIsNone(result_ex)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_cuda_available_no_local_rank(
        self,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test call when CUDA is available and no LOCAL_RANK is set"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0
        mock_device_obj = MagicMock()
        mock_device.return_value = mock_device_obj

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()
        result_state, result_ex = health_check(self.state)

        # Verify device selection
        mock_current_device.assert_called_once()
        mock_device.assert_called_once_with(0)

        # Verify thread creation and execution
        mock_thread_class.assert_called_once_with(
            target=mock_synchronize,
            args=(mock_device_obj,),
            name="CudaHealthCheckSync",
            daemon=True,
        )
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once_with(30.0)

        # Verify second synchronization
        self.assertEqual(
            mock_synchronize.call_count, 1
        )  # Called once more after thread

        self.assertEqual(result_state, self.state)
        self.assertIsNone(result_ex)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_cuda_available_with_local_rank(
        self,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test call when CUDA is available and LOCAL_RANK is set"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = "2"
        mock_device_obj = MagicMock()
        mock_device.return_value = mock_device_obj

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()
        result_state, result_ex = health_check(self.state)

        # Verify device selection with LOCAL_RANK
        mock_device.assert_called_once_with(2)

        # Verify thread creation
        mock_thread_class.assert_called_once_with(
            target=mock_synchronize,
            args=(mock_device_obj,),
            name="CudaHealthCheckSync",
            daemon=True,
        )

        self.assertEqual(result_state, self.state)
        self.assertIsNone(result_ex)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_synchronize_timeout(
        self,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test call when synchronization times out"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Thread still alive after timeout
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()

        with self.assertRaises(TimeoutError):
            health_check(self.state)

        # Verify thread was started and joined
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once_with(30.0)
        mock_thread.is_alive.assert_called_once()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_with_train_exception(
        self,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test call with train_ex parameter"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()
        train_exception = RuntimeError("Training error")

        result_state, result_ex = health_check(self.state, train_exception)

        self.assertEqual(result_state, self.state)
        self.assertEqual(result_ex, train_exception)

    @patch("torch.device")
    def test_timeout_conversion(self, mock_device):
        """Test that timeout is properly converted to seconds"""
        timeout_delta = datetime.timedelta(minutes=2, seconds=30)  # 150 seconds
        health_check = CudaHealthCheck(timeout_delta)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_initialized", return_value=True):
                with patch("os.getenv", return_value=None):
                    with patch("torch.cuda.current_device", return_value=0):
                        with patch("torch.cuda.synchronize") as mock_synchronize:
                            with patch("threading.Thread") as mock_thread_class:
                                mock_thread = MagicMock()
                                mock_thread.is_alive.return_value = False
                                mock_thread_class.return_value = mock_thread

                                health_check(self.state)

                                mock_thread.join.assert_called_once_with(150.0)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    @patch("logging.getLogger")
    def test_logging_behavior(
        self,
        mock_get_logger,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test logging behavior during health check"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0
        mock_device_obj = MagicMock()
        mock_device.return_value = mock_device_obj

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()
        health_check(self.state)

        # Verify logging calls
        self.assertEqual(mock_logger.debug.call_count, 2)
        debug_calls = mock_logger.debug.call_args_list

        # Check first debug call
        first_call_args = debug_calls[0][0][0]
        self.assertIn("1st torch.cuda.synchronize", first_call_args)

        # Check second debug call
        second_call_args = debug_calls[1][0][0]
        self.assertIn("2nd torch.cuda.synchronize", second_call_args)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("torch.cuda.synchronize")
    @patch("threading.Thread")
    @patch("os.getenv")
    @patch("logging.getLogger")
    def test_logging_timeout_scenario(
        self,
        mock_get_logger,
        mock_getenv,
        mock_thread_class,
        mock_synchronize,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test logging behavior during timeout scenario"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Timeout scenario
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()

        with self.assertRaises(TimeoutError):
            health_check(self.state)

        # Verify timeout logging
        debug_calls = mock_logger.debug.call_args_list
        self.assertEqual(len(debug_calls), 2)

        # Check timeout debug message
        timeout_call_args = debug_calls[1][0][0]
        self.assertEqual(timeout_call_args, "torch.cuda.synchronize() timed out")


class TestFaultCounter(unittest.TestCase):
    """Test the FaultCounter class"""

    def setUp(self):
        self.state = HPState()

    def test_init_with_default_max_faults(self):
        """Test FaultCounter initialization with default max_rank_faults"""
        fault_counter = FaultCounter()
        self.assertIsNone(fault_counter.max_rank_faults)
        self.assertEqual(fault_counter.faults_count, 0)

    def test_init_with_custom_max_faults(self):
        """Test FaultCounter initialization with custom max_rank_faults"""
        max_faults = 5
        fault_counter = FaultCounter(max_faults)
        self.assertEqual(fault_counter.max_rank_faults, max_faults)
        self.assertEqual(fault_counter.faults_count, 0)

    def test_call_with_no_exception(self):
        """Test call when train_ex is None"""
        fault_counter = FaultCounter(max_rank_faults=3)
        result_state, result_ex = fault_counter(self.state, None)

        self.assertEqual(result_state, self.state)
        self.assertIsNone(result_ex)
        self.assertEqual(fault_counter.faults_count, 0)

    def test_call_with_exception_under_limit(self):
        """Test call with exception when under fault limit"""
        fault_counter = FaultCounter(max_rank_faults=3)
        train_exception = RuntimeError("Training error")

        result_state, result_ex = fault_counter(self.state, train_exception)

        self.assertEqual(result_state, self.state)
        self.assertEqual(result_ex, train_exception)
        self.assertEqual(fault_counter.faults_count, 1)

    def test_call_multiple_exceptions_under_limit(self):
        """Test multiple calls with exceptions under fault limit"""
        fault_counter = FaultCounter(max_rank_faults=3)

        for i in range(3):
            train_exception = RuntimeError(f"Training error {i}")
            result_state, result_ex = fault_counter(self.state, train_exception)

            self.assertEqual(result_state, self.state)
            self.assertEqual(result_ex, train_exception)
            self.assertEqual(fault_counter.faults_count, i + 1)

    def test_call_exception_exceeds_limit(self):
        """Test call with exception that exceeds fault limit"""
        fault_counter = FaultCounter(max_rank_faults=2)

        # First two exceptions should pass
        for i in range(2):
            train_exception = RuntimeError(f"Training error {i}")
            result_state, result_ex = fault_counter(self.state, train_exception)
            self.assertEqual(fault_counter.faults_count, i + 1)

        # Third exception should raise FaultCounterExceeded
        train_exception = RuntimeError("Training error 2")
        with self.assertRaises(FaultCounterExceeded) as context:
            fault_counter(self.state, train_exception)

        # Verify exception message
        self.assertIn("faults_count=3", str(context.exception))
        self.assertIn("max_rank_faults=2", str(context.exception))
        self.assertEqual(fault_counter.faults_count, 3)

    def test_call_no_limit_set(self):
        """Test call with no fault limit set (max_rank_faults=None)"""
        fault_counter = FaultCounter(max_rank_faults=None)

        # Should handle many exceptions without raising FaultCounterExceeded
        for i in range(10):
            train_exception = RuntimeError(f"Training error {i}")
            result_state, result_ex = fault_counter(self.state, train_exception)

            self.assertEqual(result_state, self.state)
            self.assertEqual(result_ex, train_exception)
            self.assertEqual(fault_counter.faults_count, i + 1)

    def test_call_mixed_none_and_exception(self):
        """Test call with mix of None and exception values"""
        fault_counter = FaultCounter(max_rank_faults=2)

        # Call with None - should not increment counter
        result_state, result_ex = fault_counter(self.state, None)
        self.assertEqual(fault_counter.faults_count, 0)

        # Call with exception - should increment counter
        train_exception = RuntimeError("Training error")
        result_state, result_ex = fault_counter(self.state, train_exception)
        self.assertEqual(fault_counter.faults_count, 1)

        # Call with None again - should not increment counter
        result_state, result_ex = fault_counter(self.state, None)
        self.assertEqual(fault_counter.faults_count, 1)

        # Call with exception - should increment counter
        train_exception2 = ValueError("Another error")
        result_state, result_ex = fault_counter(self.state, train_exception2)
        self.assertEqual(fault_counter.faults_count, 2)

    def test_fault_counter_exceeded_exception_inheritance(self):
        """Test that FaultCounterExceeded inherits from RestartError"""
        from hyperpod_checkpointless_training.inprocess.exception import RestartError

        fault_counter = FaultCounter(max_rank_faults=0)
        train_exception = RuntimeError("Training error")

        with self.assertRaises(FaultCounterExceeded) as context:
            fault_counter(self.state, train_exception)

        # Verify inheritance
        self.assertIsInstance(context.exception, RestartError)

    def test_state_passthrough(self):
        """Test that state is passed through unchanged"""
        fault_counter = FaultCounter(max_rank_faults=5)

        # Modify state to ensure it's the same object returned
        self.state.iteration = 42
        self.state.rank = 5

        train_exception = RuntimeError("Training error")
        result_state, result_ex = fault_counter(self.state, train_exception)

        self.assertIs(result_state, self.state)
        self.assertEqual(result_state.iteration, 42)
        self.assertEqual(result_state.rank, 5)

    def test_exception_passthrough(self):
        """Test that train_ex is passed through unchanged"""
        fault_counter = FaultCounter(max_rank_faults=5)

        original_exception = RuntimeError("Original error")
        result_state, result_ex = fault_counter(self.state, original_exception)

        self.assertIs(result_ex, original_exception)

    def test_zero_fault_limit(self):
        """Test behavior with zero fault limit"""
        fault_counter = FaultCounter(max_rank_faults=0)

        # First exception should immediately exceed limit
        train_exception = RuntimeError("Training error")
        with self.assertRaises(FaultCounterExceeded):
            fault_counter(self.state, train_exception)

        self.assertEqual(fault_counter.faults_count, 1)

    def test_fault_counter_persistence(self):
        """Test that fault counter persists across calls"""
        fault_counter = FaultCounter(max_rank_faults=3)

        # Create multiple exceptions and verify counter persistence
        exceptions = [
            RuntimeError("Error 1"),
            ValueError("Error 2"),
            TypeError("Error 3"),
        ]

        for i, exception in enumerate(exceptions):
            result_state, result_ex = fault_counter(self.state, exception)
            self.assertEqual(fault_counter.faults_count, i + 1)
            self.assertEqual(result_ex, exception)

        # Next exception should exceed limit
        with self.assertRaises(FaultCounterExceeded):
            fault_counter(self.state, RuntimeError("Final error"))


class TestHealthCheckIntegration(unittest.TestCase):
    """Integration tests for health check classes"""

    def setUp(self):
        self.state = HPState()

    def test_cuda_health_check_integration_no_cuda(self):
        """Integration test for CudaHealthCheck when CUDA is not available"""
        health_check = CudaHealthCheck(timeout=datetime.timedelta(seconds=1))

        with patch("torch.cuda.is_available", return_value=False):
            result_state, result_ex = health_check(self.state)

            self.assertEqual(result_state, self.state)
            self.assertIsNone(result_ex)

    def test_fault_counter_integration_workflow(self):
        """Integration test for FaultCounter workflow"""
        fault_counter = FaultCounter(max_rank_faults=2)

        # Simulate a workflow with multiple health checks
        exceptions = [
            None,  # No error
            RuntimeError("First error"),  # First fault
            None,  # No error
            ValueError("Second error"),  # Second fault
            TypeError("Third error"),  # Should exceed limit
        ]

        for i, exception in enumerate(exceptions[:-1]):
            result_state, result_ex = fault_counter(self.state, exception)
            self.assertEqual(result_state, self.state)
            self.assertEqual(result_ex, exception)

        # Last exception should raise FaultCounterExceeded
        with self.assertRaises(FaultCounterExceeded):
            fault_counter(self.state, exceptions[-1])

    def test_combined_health_checks_simulation(self):
        """Simulate using multiple health checks together"""
        cuda_health_check = CudaHealthCheck(timeout=datetime.timedelta(seconds=1))
        fault_counter = FaultCounter(max_rank_faults=1)

        # Simulate CUDA health check passing
        with patch("torch.cuda.is_available", return_value=False):
            state, exception = cuda_health_check(self.state, None)

        # Then run fault counter
        state, exception = fault_counter(state, exception)

        self.assertEqual(state, self.state)
        self.assertIsNone(exception)
        self.assertEqual(fault_counter.faults_count, 0)


class TestHealthCheckEdgeCases(unittest.TestCase):
    """Edge case tests for health check classes"""

    def setUp(self):
        self.state = HPState()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("os.getenv")
    def test_cuda_health_check_invalid_local_rank(
        self, mock_getenv, mock_is_initialized, mock_is_available
    ):
        """Test CudaHealthCheck with invalid LOCAL_RANK value"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = "invalid"

        health_check = CudaHealthCheck()

        # Should raise ValueError when trying to convert "invalid" to int
        with self.assertRaises(ValueError):
            health_check(self.state)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("torch.cuda.current_device")
    @patch("torch.device")
    @patch("threading.Thread")
    @patch("os.getenv")
    def test_cuda_health_check_second_sync_exception(
        self,
        mock_getenv,
        mock_thread_class,
        mock_device,
        mock_current_device,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test CudaHealthCheck when second synchronization raises exception"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = None
        mock_current_device.return_value = 0

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        health_check = CudaHealthCheck()

        # Patch the second synchronize call to raise an exception
        with patch("torch.cuda.synchronize", side_effect=RuntimeError("CUDA error")):
            with self.assertRaises(RuntimeError):
                health_check(self.state)

    @patch("torch.device")
    def test_cuda_health_check_very_short_timeout(self, mock_device):
        """Test CudaHealthCheck with very short timeout"""
        short_timeout = datetime.timedelta(microseconds=1)
        health_check = CudaHealthCheck(short_timeout)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_initialized", return_value=True):
                with patch("os.getenv", return_value=None):
                    with patch("torch.cuda.current_device", return_value=0):
                        with patch("torch.cuda.synchronize"):
                            with patch("threading.Thread") as mock_thread_class:
                                mock_thread = MagicMock()
                                mock_thread.is_alive.return_value = (
                                    True  # Likely to timeout
                                )
                                mock_thread_class.return_value = mock_thread

                                with self.assertRaises(TimeoutError):
                                    health_check(self.state)

    @patch("torch.device")
    def test_cuda_health_check_zero_timeout(self, mock_device):
        """Test CudaHealthCheck with zero timeout"""
        zero_timeout = datetime.timedelta(seconds=0)
        health_check = CudaHealthCheck(zero_timeout)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_initialized", return_value=True):
                with patch("os.getenv", return_value=None):
                    with patch("torch.cuda.current_device", return_value=0):
                        with patch("torch.cuda.synchronize"):
                            with patch("threading.Thread") as mock_thread_class:
                                mock_thread = MagicMock()
                                mock_thread.is_alive.return_value = False
                                mock_thread_class.return_value = mock_thread

                                result_state, result_ex = health_check(self.state)

                                mock_thread.join.assert_called_once_with(0.0)
                                self.assertEqual(result_state, self.state)
                                self.assertIsNone(result_ex)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.is_initialized")
    @patch("os.getenv")
    def test_cuda_health_check_negative_local_rank(
        self, mock_getenv, mock_is_initialized, mock_is_available
    ):
        """Test CudaHealthCheck with negative LOCAL_RANK value"""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_getenv.return_value = "-1"

        with patch("torch.device") as mock_device:
            with patch("torch.cuda.synchronize"):
                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = MagicMock()
                    mock_thread.is_alive.return_value = False
                    mock_thread_class.return_value = mock_thread

                    health_check = CudaHealthCheck()
                    result_state, result_ex = health_check(self.state)

                    # Should still work with negative device ID
                    mock_device.assert_called_once_with(-1)
                    self.assertEqual(result_state, self.state)
                    self.assertIsNone(result_ex)

    def test_fault_counter_large_number_of_faults(self):
        """Test FaultCounter with large number of faults"""
        large_limit = 1000
        fault_counter = FaultCounter(max_rank_faults=large_limit)

        # Add many faults up to the limit
        for i in range(large_limit):
            exception = RuntimeError(f"Error {i}")
            result_state, result_ex = fault_counter(self.state, exception)
            self.assertEqual(fault_counter.faults_count, i + 1)

        # Next fault should exceed the limit
        with self.assertRaises(FaultCounterExceeded):
            fault_counter(self.state, RuntimeError("Final error"))

    def test_fault_counter_different_exception_types(self):
        """Test FaultCounter with different exception types"""
        fault_counter = FaultCounter(max_rank_faults=5)

        exception_types = [
            RuntimeError("Runtime error"),
            ValueError("Value error"),
            TypeError("Type error"),
            KeyError("Key error"),
            AttributeError("Attribute error"),
        ]

        for i, exception in enumerate(exception_types):
            result_state, result_ex = fault_counter(self.state, exception)
            self.assertEqual(result_state, self.state)
            self.assertEqual(result_ex, exception)
            self.assertEqual(fault_counter.faults_count, i + 1)

    def test_fault_counter_exception_message_format(self):
        """Test FaultCounterExceeded exception message format"""
        fault_counter = FaultCounter(max_rank_faults=1)

        # First exception should pass
        fault_counter(self.state, RuntimeError("First error"))

        # Second exception should raise FaultCounterExceeded with specific format
        with self.assertRaises(FaultCounterExceeded) as context:
            fault_counter(self.state, RuntimeError("Second error"))

        exception_message = str(context.exception)
        self.assertIn("faults_count=2", exception_message)
        self.assertIn("max_rank_faults=1", exception_message)

    @patch("torch.device")
    def test_cuda_health_check_thread_naming(self, mock_device):
        """Test that CUDA health check thread has correct name"""
        health_check = CudaHealthCheck()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_initialized", return_value=True):
                with patch("os.getenv", return_value=None):
                    with patch("torch.cuda.current_device", return_value=0):
                        with patch("torch.cuda.synchronize"):
                            with patch("threading.Thread") as mock_thread_class:
                                mock_thread = MagicMock()
                                mock_thread.is_alive.return_value = False
                                mock_thread_class.return_value = mock_thread

                                health_check(self.state)

                                # Verify thread name
                                call_kwargs = mock_thread_class.call_args[1]
                                self.assertEqual(
                                    call_kwargs["name"], "CudaHealthCheckSync"
                                )
                                self.assertTrue(call_kwargs["daemon"])

    def test_cuda_health_check_multiple_calls_same_instance(self):
        """Test calling the same CudaHealthCheck instance multiple times"""
        health_check = CudaHealthCheck()

        with patch("torch.cuda.is_available", return_value=False):
            # Call multiple times
            for i in range(3):
                result_state, result_ex = health_check(self.state)
                self.assertEqual(result_state, self.state)
                self.assertIsNone(result_ex)

    def test_fault_counter_reset_behavior(self):
        """Test that FaultCounter doesn't reset between calls"""
        fault_counter = FaultCounter(max_rank_faults=3)

        # Add some faults
        fault_counter(self.state, RuntimeError("Error 1"))
        fault_counter(self.state, RuntimeError("Error 2"))
        self.assertEqual(fault_counter.faults_count, 2)

        # Call with None - should not reset counter
        fault_counter(self.state, None)
        self.assertEqual(fault_counter.faults_count, 2)

        # Add another fault - should still be at 3
        fault_counter(self.state, RuntimeError("Error 3"))
        self.assertEqual(fault_counter.faults_count, 3)

    def test_health_check_state_modification(self):
        """Test that health checks don't modify the state object"""
        original_state = HPState()
        original_state.iteration = 100
        original_state.rank = 5

        # Test CudaHealthCheck
        cuda_health_check = CudaHealthCheck()
        with patch("torch.cuda.is_available", return_value=False):
            result_state, _ = cuda_health_check(original_state)
            self.assertIs(result_state, original_state)
            self.assertEqual(result_state.iteration, 100)
            self.assertEqual(result_state.rank, 5)

        # Test FaultCounter
        fault_counter = FaultCounter(max_rank_faults=5)
        result_state, _ = fault_counter(original_state, RuntimeError("Test error"))
        self.assertIs(result_state, original_state)
        self.assertEqual(result_state.iteration, 100)
        self.assertEqual(result_state.rank, 5)


if __name__ == "__main__":
    unittest.main()
