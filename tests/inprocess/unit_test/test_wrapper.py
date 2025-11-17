import os
import threading
import time
import unittest
from functools import partial
from unittest.mock import MagicMock, Mock, call, patch

from hyperpod_checkpointless_training.inprocess.exception import (
    HealthCheckError,
    InternalError,
    RankShouldRestart,
)
from hyperpod_checkpointless_training.inprocess.utils import AtomicInt

# Import your classes
from hyperpod_checkpointless_training.inprocess.wrap import HPCallWrapper, HPWrapper


class TestHPWrapper(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_abort = Mock(spec=type(None))
        self.mock_finalize = Mock(spec=type(None))
        self.mock_health_check = Mock()
        self.mock_hp_api = Mock()
        self.mock_hp_api_factory = Mock(return_value=self.mock_hp_api)

        # Setup distributed environment mock
        patch("torch.distributed.is_available", return_value=True).start()

    def tearDown(self):
        patch.stopall()

    def test_wrapper_initialization(self):
        """Test if HPWrapper initializes correctly with default parameters"""
        wrapper = HPWrapper(
            abort=self.mock_abort,
            finalize=self.mock_finalize,
            health_check=self.mock_health_check,
            hp_api_factory=self.mock_hp_api_factory,
        )

        self.assertEqual(wrapper.abort, self.mock_abort)
        self.assertEqual(wrapper.finalize, self.mock_finalize)
        self.assertEqual(wrapper.health_check, self.mock_health_check)
        self.assertEqual(wrapper.hp_api, self.mock_hp_api)
        self.assertIsInstance(wrapper.seq, AtomicInt)
        self.assertEqual(wrapper.seq.i, -1)
        self.assertIsNone(wrapper.abort_timeout)
        self.assertIsNone(wrapper.trace_file_path)
        self.assertTrue(wrapper.async_raise_before_abort)
        self.assertTrue(wrapper.enabled)
        self.assertFalse(wrapper.early_abort_communicator)
        self.assertIsNone(wrapper.checkpoint_manager)
        self.assertTrue(wrapper.check_memory_status)

    def test_wrapper_initialization_with_all_params(self):
        """Test HPWrapper initialization with all parameters"""
        mock_checkpoint_manager = Mock()
        wrapper = HPWrapper(
            abort=self.mock_abort,
            finalize=self.mock_finalize,
            health_check=self.mock_health_check,
            hp_api_factory=self.mock_hp_api_factory,
            abort_timeout=30.0,
            enabled=True,
            trace_file_path="/tmp/trace",
            async_raise_before_abort=False,
            early_abort_communicator=True,
            checkpoint_manager=mock_checkpoint_manager,
            check_memory_status=False,
        )

        self.assertEqual(wrapper.abort, self.mock_abort)
        self.assertEqual(wrapper.finalize, self.mock_finalize)
        self.assertEqual(wrapper.health_check, self.mock_health_check)
        self.assertEqual(wrapper.hp_api, self.mock_hp_api)
        self.assertEqual(wrapper.abort_timeout, 30.0)
        self.assertEqual(wrapper.trace_file_path, "/tmp/trace")
        self.assertEqual(wrapper.checkpoint_manager, mock_checkpoint_manager)
        self.assertFalse(wrapper.async_raise_before_abort)
        self.assertTrue(wrapper.early_abort_communicator)
        self.assertFalse(wrapper.check_memory_status)

    def test_wrapper_torch_distributed_not_available(self):
        """Test wrapper when torch distributed is not available"""
        with patch("torch.distributed.is_available", return_value=False):
            with self.assertRaises(ValueError):
                HPWrapper(hp_api_factory=self.mock_hp_api_factory)

    def test_wrapper_disabled(self):
        """Test if wrapper behaves correctly when disabled"""
        wrapper = HPWrapper(enabled=False)

        # Create a mock function
        mock_fn = Mock()

        # When wrapper is disabled, it should return the original function
        wrapped_fn = wrapper(mock_fn)
        self.assertEqual(wrapped_fn, mock_fn)

    def test_wrapper_enabled_call(self):
        """Test wrapper when enabled returns wrapped function"""
        wrapper = HPWrapper(
            abort=self.mock_abort,
            finalize=self.mock_finalize,
            health_check=self.mock_health_check,
            hp_api_factory=self.mock_hp_api_factory,
        )

        def test_fn():
            return "test"

        wrapped_fn = wrapper(test_fn)
        self.assertNotEqual(wrapped_fn, test_fn)
        self.assertEqual(wrapped_fn.__name__, test_fn.__name__)

    def test_wrapped_function_execution(self):
        """Test the wrapped function execution path"""
        wrapper = HPWrapper(
            abort=self.mock_abort,
            finalize=self.mock_finalize,
            health_check=self.mock_health_check,
            hp_api_factory=self.mock_hp_api_factory,
        )

        def test_fn(x, y=10):
            return x + y

        wrapped_fn = wrapper(test_fn)

        # Mock the HPCallWrapper context manager
        with patch("hyperpod_checkpointless_training.inprocess.wrap.HPCallWrapper") as mock_call_wrapper_class:
            mock_call_wrapper = Mock()
            mock_call_wrapper.__enter__ = Mock(return_value=mock_call_wrapper)
            mock_call_wrapper.__exit__ = Mock(return_value=None)
            mock_call_wrapper.return_value = 15
            mock_call_wrapper_class.return_value = mock_call_wrapper

            result = wrapped_fn(5, y=10)

            # Verify the call wrapper was created and called
            mock_call_wrapper_class.assert_called_once_with(wrapper)
            mock_call_wrapper.assert_called_once_with(test_fn, 5, y=10)


class TestHPCallWrapper(unittest.TestCase):
    def setUp(self):
        os.environ["RANK"] = "0"
        os.environ["HPWRAPPER_LOG_LEVEL"] = "debug"  # test with full logging
        # Mock dependencies
        self.mock_wrapper = Mock()
        self.mock_wrapper.hp_api = Mock()
        self.mock_wrapper.hp_api.hyperpod_send = Mock()
        self.mock_wrapper.hp_api.hyperpod_barrier = Mock(return_value=1)
        self.mock_wrapper.abort = Mock()
        self.mock_wrapper.finalize = Mock()
        self.mock_wrapper.health_check = Mock()
        self.mock_wrapper.seq = AtomicInt(-1)
        self.mock_wrapper.abort_timeout = None
        self.mock_wrapper.trace_file_path = None
        self.mock_wrapper.async_raise_before_abort = True
        self.mock_wrapper.check_memory_status = False
        self.mock_wrapper.checkpoint_manager = Mock()
        self.mock_wrapper.early_abort_communicator = False

    def patch_wrapper(self, call_wrapper):
        def dummy_create_start_hp_fault_handling_thread(self):
            self.hp_fault_handling_thread = Mock()

        def dummy_create_start_hp_monitor_thread(self):
            self.hp_monitor_thread = Mock()

        call_wrapper.start_hp_fault_handling_thread = partial(
            dummy_create_start_hp_fault_handling_thread, call_wrapper
        )
        call_wrapper.start_hp_monitor_thread = partial(
            dummy_create_start_hp_monitor_thread, call_wrapper
        )

    def test_step_upon_restart_initialization(self):
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        self.assertEqual(call_wrapper.step_upon_restart, 0)

    def test_successful_function_execution(self):
        """Test successful execution of a wrapped function"""

        call_wrapper = HPCallWrapper(self.mock_wrapper)

        self.patch_wrapper(call_wrapper)

        # Test function
        def test_fn():
            return "success"

        # Execute the function
        result = call_wrapper(test_fn)
        self.assertEqual(result, "success")

    def test_failure_function_execution(self):
        """Test successful execution of a wrapped function"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        self.mock_wrapper.trace_file_path = None

        self.patch_wrapper(call_wrapper)

        def track_calls(func):
            def wrapper(*args, **kwargs):
                wrapper.call_count += 1
                return func(wrapper=wrapper)

            wrapper.call_count = 0
            return wrapper

        # Test function
        @track_calls
        def test_fn(wrapper):
            print(wrapper.call_count)
            if wrapper.call_count == 1:
                raise RankShouldRestart("First run failure")
            else:
                return "success"

        # Execute the function
        result = call_wrapper(test_fn)
        self.assertEqual(result, "success")

    def test_failure_function_execution_with_trace(self):
        """Test successful execution of a wrapped function"""
        self.mock_wrapper.trace_file_path = "/tmp/profile.json"
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        self.patch_wrapper(call_wrapper)

        def track_calls(func):
            def wrapper(*args, **kwargs):
                wrapper.call_count += 1
                return func(wrapper=wrapper)

            wrapper.call_count = 0
            return wrapper

        # Test function
        @track_calls
        def test_fn(wrapper):
            print(wrapper.call_count)
            if wrapper.call_count == 1:
                raise RankShouldRestart("First run failure")
            else:
                return "success"

        # Execute the function
        result = call_wrapper(test_fn)
        self.assertEqual(result, "success")

    def test_handle_finalize(self):
        """Test finalize handling"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_exception = Exception("Test exception")

        call_wrapper.handle_finalize(mock_exception)
        self.mock_wrapper.finalize.assert_called_once()

    def test_handle_fn_exception(self):
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_exception = Exception("Test exception")
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.hp_fault_handling_thread.join = Mock()
        call_wrapper.hp_fault_handling_thread.shutdown = Mock()

        with self.assertRaises(RankShouldRestart):
            call_wrapper.handle_fn_exception(mock_exception)

    def test_handle_health_check(self):
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_exception = Exception("Test exception")

        call_wrapper.handle_health_check(mock_exception)
        self.mock_wrapper.health_check.assert_called_once()

    def test_initialization_with_trace_file(self):
        """Test HPCallWrapper initialization with trace file"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        self.assertTrue(call_wrapper.tracing_enabled)
        self.assertIsNotNone(call_wrapper.tracer)
        self.assertFalse(call_wrapper.is_tracing)

    def test_initialization_without_trace_file(self):
        """Test HPCallWrapper initialization without trace file"""
        self.mock_wrapper.trace_file_path = None
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        self.assertFalse(call_wrapper.tracing_enabled)
        self.assertIsNone(call_wrapper.tracer)
        self.assertFalse(call_wrapper.is_tracing)

    @patch.dict(os.environ, {"RANK": "0"})
    def test_enable_tracing_rank_0(self):
        """Test enable_tracing for rank 0"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 0

        with patch.object(call_wrapper.tracer, "start") as mock_start:
            call_wrapper.enable_tracing()
            mock_start.assert_called_once()
            self.assertTrue(call_wrapper.is_tracing)

    @patch.dict(os.environ, {"RANK": "1"})
    def test_enable_tracing_rank_not_0(self):
        """Test enable_tracing for rank != 0"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 1

        with patch.object(call_wrapper.tracer, "start") as mock_start:
            call_wrapper.enable_tracing()
            mock_start.assert_not_called()
            self.assertFalse(call_wrapper.is_tracing)

    def test_enable_tracing_already_tracing(self):
        """Test enable_tracing when already tracing"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 0
        call_wrapper.is_tracing = True

        with patch.object(call_wrapper.tracer, "start") as mock_start:
            call_wrapper.enable_tracing()
            mock_start.assert_not_called()

    def test_enable_tracing_disabled(self):
        """Test enable_tracing when tracing is disabled"""
        self.mock_wrapper.trace_file_path = None
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 0

        call_wrapper.enable_tracing()
        self.assertFalse(call_wrapper.is_tracing)

    @patch("time.monotonic", return_value=123.456)
    def test_end_tracing_rank_0(self, mock_time):
        """Test end_tracing for rank 0"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 0
        call_wrapper.is_tracing = True

        with (
            patch.object(call_wrapper.tracer, "stop") as mock_stop,
            patch.object(call_wrapper.tracer, "save") as mock_save,
        ):
            call_wrapper.end_tracing()
            mock_stop.assert_called_once()
            mock_save.assert_called_once_with("/tmp/trace/profile_123.456.json")
            self.assertFalse(call_wrapper.is_tracing)

    def test_end_tracing_not_tracing(self):
        """Test end_tracing when not tracing"""
        self.mock_wrapper.trace_file_path = "/tmp/trace"
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.state.rank = 0
        call_wrapper.is_tracing = False

        with patch.object(call_wrapper.tracer, "stop") as mock_stop:
            call_wrapper.end_tracing()
            mock_stop.assert_not_called()

    def test_initialize_barrier_success(self):
        """Test successful initialize_barrier"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        self.patch_wrapper(call_wrapper)

        with patch.object(call_wrapper.state, "get_distributed_vars") as mock_get_vars:
            call_wrapper.initialize_barrier()

            # Verify barrier was called
            self.mock_wrapper.hp_api.hyperpod_barrier.assert_called_once_with(
                call_wrapper.state.rank
            )
            # Verify distributed vars were retrieved twice
            self.assertEqual(mock_get_vars.call_count, 2)
            # Verify fault handling thread was started
            self.assertIsNotNone(call_wrapper.hp_fault_handling_thread)

    def test_initialize_barrier_exception(self):
        """Test initialize_barrier with exception"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        self.patch_wrapper(call_wrapper)

        # Make hyperpod_barrier raise an exception
        self.mock_wrapper.hp_api.hyperpod_barrier.side_effect = Exception(
            "Barrier failed"
        )

        with self.assertRaises(Exception):
            call_wrapper.initialize_barrier()

    def test_handle_fn_exception_step_greater_than_1(self):
        """Test handle_fn_exception when step_upon_restart > 1"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.step_upon_restart = 2
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.hp_fault_handling_thread.join = Mock()
        call_wrapper.hp_fault_handling_thread.shutdown = Mock()

        mock_exception = Exception("Test exception")

        with self.assertRaises(RankShouldRestart):
            call_wrapper.handle_fn_exception(mock_exception)

        # Verify hyperpod_send was called without plr_restart
        self.mock_wrapper.hp_api.hyperpod_send.assert_called_once()
        args, kwargs = self.mock_wrapper.hp_api.hyperpod_send.call_args
        self.assertNotIn("plr_restart", kwargs)

    def test_handle_fn_exception_step_1(self):
        """Test handle_fn_exception when step_upon_restart <= 1"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.step_upon_restart = 1
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.hp_fault_handling_thread.join = Mock()
        call_wrapper.hp_fault_handling_thread.shutdown = Mock()

        mock_exception = Exception("Test exception")

        with self.assertRaises(RankShouldRestart):
            call_wrapper.handle_fn_exception(mock_exception)

        # Verify hyperpod_send was called with plr_restart=True
        self.mock_wrapper.hp_api.hyperpod_send.assert_called_once()
        args, kwargs = self.mock_wrapper.hp_api.hyperpod_send.call_args
        self.assertTrue(kwargs.get("plr_restart", False))

    def test_handle_fn_exception_internal_error(self):
        """Test handle_fn_exception when internal exception occurs"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.step_upon_restart = 2
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.hp_fault_handling_thread.join.side_effect = Exception(
            "Join failed"
        )

        mock_exception = Exception("Test exception")

        with self.assertRaises(InternalError):
            call_wrapper.handle_fn_exception(mock_exception)

    def test_handle_finalize_no_finalize(self):
        """Test handle_finalize when finalize is None"""
        self.mock_wrapper.finalize = None
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        mock_exception = RankShouldRestart()
        call_wrapper.handle_finalize(mock_exception)
        # Should not raise any exception

    def test_handle_finalize_with_exception(self):
        """Test handle_finalize when finalize raises exception"""
        self.mock_wrapper.finalize.side_effect = Exception("Finalize failed")
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        mock_exception = RankShouldRestart()

        with self.assertRaises(Exception):
            call_wrapper.handle_finalize(mock_exception)

    def test_handle_health_check_no_health_check(self):
        """Test handle_health_check when health_check is None"""
        self.mock_wrapper.health_check = None
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        mock_exception = RankShouldRestart()
        call_wrapper.handle_health_check(mock_exception)
        # Should not raise any exception

    def test_handle_health_check_with_exception(self):
        """Test handle_health_check when health_check raises exception"""
        self.mock_wrapper.health_check.side_effect = Exception("Health check failed")
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        mock_exception = RankShouldRestart()

        with self.assertRaises(HealthCheckError):
            call_wrapper.handle_health_check(mock_exception)

    @patch("gc.collect")
    def test_handle_gc_success(self, mock_collect):
        """Test handle_gc successful garbage collection"""
        mock_collect.side_effect = [5, 3, 0]  # Simulate decreasing garbage
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        call_wrapper.handle_gc()
        self.assertEqual(mock_collect.call_count, 3)

    @patch("gc.collect")
    def test_handle_gc_exception(self, mock_collect):
        """Test handle_gc with exception"""
        mock_collect.side_effect = Exception("GC failed")
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        # Should not raise exception, just log it
        call_wrapper.handle_gc()
        
    @patch("hyperpod_checkpointless_training.inprocess.wrap.memory_status")
    def test_check_memory_status_enabled(self, mock_memory_status):
        """Test memory status check during restart when enabled"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.check_memory_status = True
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.checkpoint_manager = Mock()
        mock_memory_status.return_value = ("Memory info", {})

        mock_exception = RankShouldRestart()

        with (
            patch.object(call_wrapper, "enable_tracing"),
            patch.object(call_wrapper, "shutdown_hp_fault_handling_thread"),
            patch.object(call_wrapper, "handle_finalize"),
            patch.object(call_wrapper, "handle_gc"),
            patch.object(call_wrapper, "handle_health_check"),
        ):
            call_wrapper.restart(mock_exception)
            mock_memory_status.assert_has_calls([
                call(tag="Before Offload"),
                call(tag="After restart")
            ])

    @patch("hyperpod_checkpointless_training.inprocess.wrap.memory_status")
    def test_check_memory_status_disabled(self, mock_memory_status):
        """Test memory status check during restart when disabled"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.check_memory_status = False
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.checkpoint_manager = Mock()

        mock_exception = RankShouldRestart()

        with (
            patch.object(call_wrapper, "enable_tracing"),
            patch.object(call_wrapper, "shutdown_hp_fault_handling_thread"),
            patch.object(call_wrapper, "handle_finalize"),
            patch.object(call_wrapper, "handle_gc"),
            patch.object(call_wrapper, "handle_health_check"),
        ):
            call_wrapper.restart(mock_exception)
            mock_memory_status.assert_not_called()

    def test_handle_unknown_error(self):
        """Test handle_unknown_error"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_exception = Exception("Unknown error")

        with self.assertRaises(InternalError):
            call_wrapper.handle_unknown_error(mock_exception)

    def test_offload_after_restart(self):
        """Test that checkpoint offloading occurs after restart sequence"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.atomic_lock = Mock()
        call_wrapper.check_memory_status = True
        
        mock_exception = RankShouldRestart()
        
        with (
            patch.object(call_wrapper, "enable_tracing"),
            patch.object(call_wrapper, "shutdown_hp_fault_handling_thread"),
            patch.object(call_wrapper, "handle_finalize"),
            patch.object(call_wrapper, "handle_gc"),
            patch.object(call_wrapper, "handle_health_check"),
            patch("hyperpod_checkpointless_training.inprocess.wrap.memory_status") as mock_memory_status,
        ):
            mock_memory_status.return_value = ("Memory info", {})
            
            call_wrapper.restart(mock_exception)
            
            # Verify the sequence of operations
            self.mock_wrapper.checkpoint_manager.reset_checkpointless_recovery_validation.assert_called_once()
            call_wrapper.atomic_lock.force_release.assert_called_once()
            
            # Verify memory status was checked before offload
            mock_memory_status.assert_has_calls([
                call(tag="Before Offload"),
                call(tag="After restart")
            ])
            
            # Verify checkpoint offload was called
            self.mock_wrapper.checkpoint_manager.maybe_offload_checkpoint.assert_called_once()

    def test_offload_after_restart_memory_check_disabled(self):
        """Test restart sequence when memory checks are disabled"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.atomic_lock = Mock()
        call_wrapper.check_memory_status = False
        
        mock_exception = RankShouldRestart()
        
        with (
            patch.object(call_wrapper, "enable_tracing"),
            patch.object(call_wrapper, "shutdown_hp_fault_handling_thread"),
            patch.object(call_wrapper, "handle_finalize"),
            patch.object(call_wrapper, "handle_gc"),
            patch.object(call_wrapper, "handle_health_check"),
            patch("hyperpod_checkpointless_training.inprocess.wrap.memory_status") as mock_memory_status,
        ):
            call_wrapper.restart(mock_exception)
            
            # Verify core operations still occur
            self.mock_wrapper.checkpoint_manager.reset_checkpointless_recovery_validation.assert_called_once()
            call_wrapper.atomic_lock.force_release.assert_called_once()
            
            # Verify no memory status checks occurred
            mock_memory_status.assert_not_called()
            
            # Verify checkpoint offload still occurs
            self.mock_wrapper.checkpoint_manager.maybe_offload_checkpoint.assert_called_once()

    def test_restart_step_greater_than_1(self):
        """Test restart when step_upon_restart > 1"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.step_upon_restart = 2
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.checkpoint_manager = Mock()

        mock_exception = RankShouldRestart()

        with (
            patch.object(call_wrapper, "enable_tracing") as mock_enable_tracing,
            patch.object(
                call_wrapper, "shutdown_hp_fault_handling_thread"
            ) as mock_shutdown,
            patch.object(call_wrapper, "handle_finalize") as mock_finalize,
            patch.object(call_wrapper, "handle_gc") as mock_gc,
            patch.object(call_wrapper, "handle_health_check") as mock_health,
        ):
            call_wrapper.restart(mock_exception)

            mock_enable_tracing.assert_called_once()
            mock_shutdown.assert_called_once()
            mock_finalize.assert_called_once_with(mock_exception)
            mock_gc.assert_called_once()
            mock_health.assert_called_once_with(mock_exception)
            call_wrapper.checkpoint_manager.reset_checkpointless_recovery_validation.assert_called_once()

            # Should not call hyperpod_send for step > 1
            self.mock_wrapper.hp_api.hyperpod_send.assert_not_called()

    def test_restart_step_1(self):
        """Test restart when step_upon_restart <= 1"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.step_upon_restart = 1
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.checkpoint_manager = Mock()

        mock_exception = RankShouldRestart()

        with (
            patch.object(call_wrapper, "enable_tracing"),
            patch.object(call_wrapper, "shutdown_hp_fault_handling_thread"),
            patch.object(call_wrapper, "handle_finalize"),
            patch.object(call_wrapper, "handle_gc"),
            patch.object(call_wrapper, "handle_health_check"),
        ):
            call_wrapper.restart(mock_exception)

            # Should call hyperpod_send with plr_restart=True for step <= 1
            self.mock_wrapper.hp_api.hyperpod_send.assert_called_once()
            args, kwargs = self.mock_wrapper.hp_api.hyperpod_send.call_args
            self.assertTrue(kwargs.get("plr_restart", False))
            call_wrapper.checkpoint_manager.reset_checkpointless_recovery_validation.assert_called_once()

    @patch("hyperpod_checkpointless_training.inprocess.wrap.substitute_param_value")
    def test_launch_success(self, mock_substitute):
        """Test successful launch"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_substitute.return_value = ((), {})

        def test_fn():
            return "success"

        result = call_wrapper.launch(test_fn)
        self.assertEqual(result, "success")
        mock_substitute.assert_called_once()

    @patch("hyperpod_checkpointless_training.inprocess.wrap.substitute_param_value")
    @patch("traceback.print_exc")
    @patch("traceback.format_exception")
    def test_launch_with_exception(
        self, mock_format_exc, mock_print_exc, mock_substitute
    ):
        """Test launch with exception"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_substitute.return_value = ((), {})
        mock_format_exc.return_value = ["Traceback", "Exception: Test error"]

        def test_fn():
            raise Exception("Test error")

        with (
            patch.object(call_wrapper, "enable_tracing") as mock_enable_tracing,
            patch.object(call_wrapper, "handle_fn_exception") as mock_handle_exception,
        ):
            call_wrapper.launch(test_fn)

            mock_enable_tracing.assert_called_once()
            mock_handle_exception.assert_called_once()
            mock_print_exc.assert_called_once()
            mock_format_exc.assert_called_once()

    def test_run_success(self):
        """Test successful run"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        def test_fn():
            return "success"

        with (
            patch.object(call_wrapper, "initialize_barrier") as mock_init_barrier,
            patch.object(call_wrapper, "end_tracing") as mock_end_tracing,
            patch.object(call_wrapper, "launch") as mock_launch,
        ):
            mock_launch.return_value = "success"
            result = call_wrapper.run(test_fn)

            self.assertEqual(result, "success")
            mock_init_barrier.assert_called_once()
            mock_end_tracing.assert_called_once()
            mock_launch.assert_called_once_with(test_fn)

    def test_run_with_restart(self):
        """Test run with RankShouldRestart exception"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        def test_fn():
            return "success"

        call_count = 0

        def mock_launch(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RankShouldRestart("Restart needed")
            return "success"

        with (
            patch.object(call_wrapper, "initialize_barrier"),
            patch.object(call_wrapper, "end_tracing"),
            patch.object(call_wrapper, "launch", side_effect=mock_launch),
            patch.object(call_wrapper, "restart") as mock_restart,
        ):
            result = call_wrapper.run(test_fn)

            self.assertEqual(result, "success")
            mock_restart.assert_called_once()

    def test_start_success(self):
        """Test successful start"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        def test_fn():
            return "success"

        with (
            patch.object(call_wrapper, "run") as mock_run,
            patch.object(call_wrapper.state, "advance") as mock_advance,
        ):
            mock_run.return_value = "success"
            result = call_wrapper.start(test_fn)

            self.assertEqual(result, "success")
            mock_run.assert_called_once_with(test_fn)
            mock_advance.assert_called_once()

    def test_start_with_exception(self):
        """Test start with BaseException"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        def test_fn():
            return "success"

        with patch.object(
            call_wrapper, "run", side_effect=KeyboardInterrupt("Interrupted")
        ):
            with self.assertRaises(KeyboardInterrupt):
                call_wrapper.start(test_fn)

    def test_start_hp_monitor_thread_already_alive(self):
        """Test start_hp_monitor_thread when thread is already alive"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_monitor_thread = Mock()
        call_wrapper.hp_monitor_thread.is_alive.return_value = True

        call_wrapper.start_hp_monitor_thread()

        # Should not create a new thread
        call_wrapper.hp_monitor_thread.start.assert_not_called()

    def test_start_hp_monitor_thread_new(self):
        """Test start_hp_monitor_thread creating new thread"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_monitor_thread = None

        with patch("hyperpod_checkpointless_training.inprocess.wrap.HPMonitorThread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            call_wrapper.start_hp_monitor_thread()

            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()
            self.assertEqual(call_wrapper.hp_monitor_thread, mock_thread)

    def test_context_manager(self):
        """Test context manager functionality"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        with (
            patch.object(call_wrapper, "start_hp_monitor_thread") as mock_start,
            patch.object(call_wrapper, "shutdown") as mock_shutdown,
        ):
            with call_wrapper as wrapper:
                self.assertEqual(wrapper, call_wrapper)

            mock_start.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_shutdown_hp_fault_handling_thread_with_restart(self):
        """Test shutdown_hp_fault_handling_thread with RankShouldRestart"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_fault_handling_thread = Mock()
        call_wrapper.hp_fault_handling_thread.shutdown.side_effect = [
            RankShouldRestart(),
            None,
        ]

        with patch.object(
            call_wrapper,
            "shutdown_hp_fault_handling_thread",
            wraps=call_wrapper.shutdown_hp_fault_handling_thread,
        ) as mock_shutdown:
            call_wrapper.shutdown_hp_fault_handling_thread()
            # Should be called twice due to recursion
            self.assertEqual(mock_shutdown.call_count, 2)

    def test_shutdown_hp_monitor_thread(self):
        """Test shutdown_hp_monitor_thread"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        mock_thread = Mock()
        call_wrapper.hp_monitor_thread = mock_thread

        call_wrapper.shutdown_hp_monitor_thread()

        mock_thread.shutdown.assert_called_once()
        mock_thread.join.assert_called_once()
        self.assertIsNone(call_wrapper.hp_monitor_thread)

    def test_shutdown_hp_monitor_thread_none(self):
        """Test shutdown_hp_monitor_thread when thread is None"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)
        call_wrapper.hp_monitor_thread = None

        # Should not raise exception
        call_wrapper.shutdown_hp_monitor_thread()
        self.assertIsNone(call_wrapper.hp_monitor_thread)

    def test_start_hp_fault_handling_thread_creation(self):
        """Test start_hp_fault_handling_thread creates and starts thread"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        with patch("hyperpod_checkpointless_training.inprocess.wrap.HPFaultHandlingThread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            call_wrapper.start_hp_fault_handling_thread()

            # Verify thread was created with correct parameters
            mock_thread_class.assert_called_once_with(
                abort=self.mock_wrapper.abort,
                failure=call_wrapper.failure,
                stop_raising=call_wrapper.stop_raising,
                atomic_lock=call_wrapper.atomic_lock,
                abort_timeout=self.mock_wrapper.abort_timeout,
                async_raise_before_abort=self.mock_wrapper.async_raise_before_abort,
                seq=self.mock_wrapper.seq,
                daemon=True,
            )
            mock_thread.start.assert_called_once()
            self.assertEqual(call_wrapper.hp_fault_handling_thread, mock_thread)

    def test_shutdown_method(self):
        """Test shutdown method calls both shutdown methods"""
        call_wrapper = HPCallWrapper(self.mock_wrapper)

        with (
            patch.object(
                call_wrapper, "shutdown_hp_fault_handling_thread"
            ) as mock_shutdown_fault,
            patch.object(
                call_wrapper, "shutdown_hp_monitor_thread"
            ) as mock_shutdown_monitor,
        ):
            call_wrapper.shutdown()

            mock_shutdown_fault.assert_called_once()
            mock_shutdown_monitor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
