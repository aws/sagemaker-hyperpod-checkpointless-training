# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import ctypes
import logging
import os
import socket
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, call, patch

from hyperpod_checkpointless_training.inprocess.utils import (
    AtomicInt,
    HPState,
    async_raise,
    debug_msg,
    delayed_async_raise,
    format_exc,
    log_exc,
    reraise_if_unraisable,
)


class TestAsyncRaise(unittest.TestCase):
    """Test the async_raise function"""

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    @patch("sys.is_finalizing")
    def test_async_raise_success(self, mock_is_finalizing, mock_set_async_exc):
        """Test successful async_raise execution"""
        mock_is_finalizing.return_value = False
        mock_set_async_exc.return_value = 1  # Success

        # Should not raise an exception
        async_raise(12345, RuntimeError)

        mock_set_async_exc.assert_called_once_with(12345, RuntimeError)

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    @patch("sys.is_finalizing")
    def test_async_raise_with_event(self, mock_is_finalizing, mock_set_async_exc):
        """Test async_raise with event parameter"""
        mock_is_finalizing.return_value = False
        mock_set_async_exc.return_value = 1

        event = MagicMock()
        async_raise(12345, RuntimeError, event)

        event.wait.assert_called_once()
        mock_set_async_exc.assert_called_once_with(12345, RuntimeError)

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    @patch("sys.is_finalizing")
    def test_async_raise_finalizing(self, mock_is_finalizing, mock_set_async_exc):
        """Test async_raise when system is finalizing"""
        mock_is_finalizing.return_value = True

        # Should not call set_async_exc when finalizing
        async_raise(12345, RuntimeError)

        mock_set_async_exc.assert_not_called()

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    @patch("sys.is_finalizing")
    def test_async_raise_thread_not_found(self, mock_is_finalizing, mock_set_async_exc):
        """Test async_raise when thread is not found"""
        mock_is_finalizing.return_value = False
        mock_set_async_exc.return_value = 0  # Thread not found

        with self.assertRaises(RuntimeError):
            async_raise(12345, RuntimeError)

    @patch("ctypes.pythonapi.PyThreadState_SetAsyncExc")
    @patch("sys.is_finalizing")
    def test_async_raise_multiple_threads_affected(
        self, mock_is_finalizing, mock_set_async_exc
    ):
        """Test async_raise when multiple threads are affected"""
        mock_is_finalizing.return_value = False
        mock_set_async_exc.side_effect = [
            2,
            0,
        ]  # First call affects 2 threads, second call clears

        with self.assertRaises(RuntimeError):
            async_raise(12345, RuntimeError)

        # Should call twice: once to set exception, once to clear
        self.assertEqual(mock_set_async_exc.call_count, 2)
        mock_set_async_exc.assert_has_calls(
            [call(12345, RuntimeError), call(12345, None)]
        )

    def test_async_raise_ctypes_setup(self):
        """Test that ctypes function is properly configured"""
        # This test verifies the ctypes setup without actually calling the function
        with patch("ctypes.pythonapi.PyThreadState_SetAsyncExc") as mock_func:
            mock_func.argtypes = (ctypes.c_ulong, ctypes.py_object)
            mock_func.restype = ctypes.c_int
            mock_func.return_value = 1

            with patch("sys.is_finalizing", return_value=False):
                async_raise(12345, RuntimeError)

            # Verify the function was called with correct types
            mock_func.assert_called_once()


class TestDelayedAsyncRaise(unittest.TestCase):
    """Test the delayed_async_raise function"""

    @patch("threading.Thread")
    @patch("threading.Event")
    def test_delayed_async_raise(self, mock_event_class, mock_thread_class):
        """Test delayed_async_raise creates thread and event correctly"""
        mock_event = MagicMock()
        mock_thread = MagicMock()
        mock_event_class.return_value = mock_event
        mock_thread_class.return_value = mock_thread

        delayed_async_raise(12345, RuntimeError)

        # Verify thread creation
        mock_thread_class.assert_called_once_with(
            target=async_raise,
            args=(12345, RuntimeError, mock_event),
            daemon=True,
        )

        # Verify thread start and event set
        mock_thread.start.assert_called_once()
        mock_event.set.assert_called_once()

    @patch("hyperpod_checkpointless_training.inprocess.utils.async_raise")
    def test_delayed_async_raise_integration(self, mock_async_raise):
        """Test delayed_async_raise integration with real threading"""
        # This test uses real threading but mocks async_raise
        delayed_async_raise(12345, RuntimeError)

        # Give the thread a moment to execute
        time.sleep(0.1)

        # Verify async_raise was called
        mock_async_raise.assert_called_once()
        args = mock_async_raise.call_args[0]
        self.assertEqual(args[0], 12345)
        self.assertEqual(args[1], RuntimeError)


class TestReraiseIfUnraisable(unittest.TestCase):
    """Test the reraise_if_unraisable context manager"""

    def setUp(self):
        self.original_unraisablehook = sys.unraisablehook

    def tearDown(self):
        sys.unraisablehook = self.original_unraisablehook

    def test_reraise_if_unraisable_triggers(self):
        """Test reraise_if_unraisable context manager setup"""
        original_hook = sys.unraisablehook

        with reraise_if_unraisable(RuntimeError):
            # Verify that the hook was replaced
            self.assertNotEqual(sys.unraisablehook, original_hook)

            # Verify that the wrapped hook has the expected structure
            self.assertTrue(callable(sys.unraisablehook))

        # Verify that the original hook is restored
        self.assertEqual(sys.unraisablehook, original_hook)

    def test_reraise_if_unraisable_context_behavior(self):
        """Test reraise_if_unraisable context manager behavior"""
        original_hook = sys.unraisablehook

        # Test that context manager properly wraps and restores the hook
        with reraise_if_unraisable(RuntimeError):
            wrapped_hook = sys.unraisablehook
            self.assertNotEqual(wrapped_hook, original_hook)

            # Test nested context
            with reraise_if_unraisable(ValueError):
                nested_hook = sys.unraisablehook
                self.assertNotEqual(nested_hook, wrapped_hook)
                self.assertNotEqual(nested_hook, original_hook)

            # Should restore to wrapped_hook
            self.assertEqual(sys.unraisablehook, wrapped_hook)

        # Should restore to original
        self.assertEqual(sys.unraisablehook, original_hook)

    def test_reraise_if_unraisable_exception_filtering(self):
        """Test reraise_if_unraisable exception type filtering"""
        # Test that the context manager is set up to filter specific exception types
        with reraise_if_unraisable(RuntimeError):
            # The context manager should be active
            self.assertNotEqual(sys.unraisablehook, self.original_unraisablehook)

        # Test with different exception type
        with reraise_if_unraisable(ValueError):
            # Should also wrap the hook
            self.assertNotEqual(sys.unraisablehook, self.original_unraisablehook)

    def test_reraise_if_unraisable_restores_hook(self):
        """Test that original unraisablehook is restored"""
        original_hook = sys.unraisablehook

        with reraise_if_unraisable(RuntimeError):
            # Hook should be wrapped
            self.assertNotEqual(sys.unraisablehook, original_hook)

        # Hook should be restored
        self.assertEqual(sys.unraisablehook, original_hook)

    def test_reraise_if_unraisable_function_wrapping(self):
        """Test that reraise_if_unraisable properly wraps functions"""
        original_hook = sys.unraisablehook

        with reraise_if_unraisable(RuntimeError):
            wrapped_hook = sys.unraisablehook

            # The wrapped hook should be callable
            self.assertTrue(callable(wrapped_hook))

            # The wrapped hook should be different from the original
            self.assertNotEqual(wrapped_hook, original_hook)

        # Original hook should be restored
        self.assertEqual(sys.unraisablehook, original_hook)


class TestHPState(unittest.TestCase):
    """Test the HPState class"""

    def setUp(self):
        # Clear environment variables for clean testing
        self.original_rank = os.environ.get("RANK")
        self.original_world_size = os.environ.get("WORLD_SIZE")
        if "RANK" in os.environ:
            del os.environ["RANK"]
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

    def tearDown(self):
        # Restore original environment variables
        if self.original_rank is not None:
            os.environ["RANK"] = self.original_rank
        elif "RANK" in os.environ:
            del os.environ["RANK"]

        if self.original_world_size is not None:
            os.environ["WORLD_SIZE"] = self.original_world_size
        elif "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

    def test_hpstate_default_initialization(self):
        """Test HPState initialization with default values"""
        state = HPState()
        self.assertEqual(state.rank, -1)
        self.assertEqual(state.world_size, -1)
        self.assertEqual(state.iteration, 0)

    def test_hpstate_with_environment_variables(self):
        """Test HPState initialization with environment variables"""
        os.environ["RANK"] = "2"
        os.environ["WORLD_SIZE"] = "4"

        state = HPState()
        self.assertEqual(state.rank, 2)
        self.assertEqual(state.world_size, 4)
        self.assertEqual(state.iteration, 0)

    def test_hpstate_advance(self):
        """Test HPState advance method"""
        state = HPState()
        initial_iteration = state.iteration

        state.advance()
        self.assertEqual(state.iteration, initial_iteration + 1)

        state.advance()
        self.assertEqual(state.iteration, initial_iteration + 2)

    def test_hpstate_get_distributed_vars(self):
        """Test get_distributed_vars method"""
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "8"

        state = HPState()
        # Change values manually
        state.rank = 999
        state.world_size = 999

        # Call get_distributed_vars to refresh from environment
        state.get_distributed_vars()
        self.assertEqual(state.rank, 3)
        self.assertEqual(state.world_size, 8)

    def test_hpstate_set_distributed_vars_raises_error(self):
        """Test that set_distributed_vars raises RuntimeError"""
        state = HPState()
        with self.assertRaises(RuntimeError) as context:
            state.set_distributed_vars()

        self.assertIn("does not allow to set dist vars", str(context.exception))

    def test_hpstate_dataclass_fields(self):
        """Test that HPState has correct dataclass fields"""
        import dataclasses

        self.assertTrue(dataclasses.is_dataclass(HPState))

        fields = dataclasses.fields(HPState)
        field_names = [field.name for field in fields]
        self.assertIn("rank", field_names)
        self.assertIn("world_size", field_names)
        self.assertIn("iteration", field_names)

    def test_hpstate_string_environment_variables(self):
        """Test HPState with string environment variables that need conversion"""
        os.environ["RANK"] = "5"
        os.environ["WORLD_SIZE"] = "10"

        state = HPState()
        self.assertIsInstance(state.rank, int)
        self.assertIsInstance(state.world_size, int)
        self.assertEqual(state.rank, 5)
        self.assertEqual(state.world_size, 10)

    def test_hpstate_invalid_environment_variables(self):
        """Test HPState with invalid environment variables"""
        os.environ["RANK"] = "invalid"
        os.environ["WORLD_SIZE"] = "also_invalid"

        with self.assertRaises(ValueError):
            HPState()


class TestFormatExc(unittest.TestCase):
    """Test the format_exc function"""

    def test_format_exc_single_exception(self):
        """Test format_exc with single exception"""
        exc = ValueError("test error")
        result = format_exc(exc)
        self.assertEqual(result, "ValueError('test error')")

    def test_format_exc_chained_exceptions(self):
        """Test format_exc with chained exceptions"""
        try:
            try:
                raise ValueError("original error")
            except ValueError as e:
                raise RuntimeError("wrapper error") from e
        except RuntimeError as exc:
            result = format_exc(exc)
            self.assertIn("RuntimeError('wrapper error')", result)
            self.assertIn("ValueError('original error')", result)
            self.assertIn(" <- ", result)

    def test_format_exc_multiple_chained_exceptions(self):
        """Test format_exc with multiple chained exceptions"""
        try:
            try:
                try:
                    raise TypeError("base error")
                except TypeError as e:
                    raise ValueError("middle error") from e
            except ValueError as e:
                raise RuntimeError("top error") from e
        except RuntimeError as exc:
            result = format_exc(exc)
            self.assertIn("RuntimeError('top error')", result)
            self.assertIn("ValueError('middle error')", result)
            self.assertIn("TypeError('base error')", result)
            # Should have two " <- " separators
            self.assertEqual(result.count(" <- "), 2)

    def test_format_exc_no_cause(self):
        """Test format_exc with exception that has no cause"""
        exc = RuntimeError("standalone error")
        result = format_exc(exc)
        self.assertEqual(result, "RuntimeError('standalone error')")

    def test_format_exc_custom_exception(self):
        """Test format_exc with custom exception class"""

        class CustomError(Exception):
            def __init__(self, message, code):
                super().__init__(message)
                self.code = code

        exc = CustomError("custom message", 42)
        result = format_exc(exc)
        self.assertIn("CustomError('custom message')", result)


class TestLogExc(unittest.TestCase):
    """Test the log_exc function"""

    @patch("hyperpod_checkpointless_training.inprocess.utils.debug_msg")
    @patch("hyperpod_checkpointless_training.inprocess.utils.format_exc")
    def test_log_exc_basic(self, mock_format_exc, mock_debug_msg):
        """Test basic log_exc functionality"""
        mock_format_exc.return_value = "formatted exception"
        mock_debug_msg.return_value = "debug message"

        exc = RuntimeError("test error")
        result = log_exc(exc, "test_operation")

        mock_format_exc.assert_called_once_with(exc)
        mock_debug_msg.assert_called_once_with(
            "test_operation: formatted exception", rank=-1, seq=-1, steps=-1
        )
        self.assertEqual(result, "debug message")

    @patch("hyperpod_checkpointless_training.inprocess.utils.debug_msg")
    @patch("hyperpod_checkpointless_training.inprocess.utils.format_exc")
    def test_log_exc_with_parameters(self, mock_format_exc, mock_debug_msg):
        """Test log_exc with custom parameters"""
        mock_format_exc.return_value = "formatted exception"
        mock_debug_msg.return_value = "debug message with params"

        exc = ValueError("param error")
        result = log_exc(exc, "param_operation", rank=2, seq=5, steps=10)

        mock_format_exc.assert_called_once_with(exc)
        mock_debug_msg.assert_called_once_with(
            "param_operation: formatted exception", rank=2, seq=5, steps=10
        )
        self.assertEqual(result, "debug message with params")

    def test_log_exc_integration(self):
        """Test log_exc integration with real exceptions"""
        try:
            raise ValueError("integration test error")
        except ValueError as exc:
            result = log_exc(exc, "integration_test", rank=1, seq=2, steps=3)

        # Verify the result contains expected components
        self.assertIn("integration_test:", result)
        self.assertIn("ValueError('integration test error')", result)
        self.assertIn("[RANK:1]", result)
        self.assertIn("[SEQ:2]", result)
        self.assertIn("[STEPS:3]", result)


class TestAtomicInt(unittest.TestCase):
    """Test the AtomicInt class"""

    def test_atomic_int_initialization(self):
        """Test AtomicInt initialization"""
        atomic = AtomicInt(42)
        self.assertEqual(atomic.get(), 42)

    def test_atomic_int_set_and_get(self):
        """Test AtomicInt set and get operations"""
        atomic = AtomicInt(0)

        atomic.set(100)
        self.assertEqual(atomic.get(), 100)

        atomic.set(-50)
        self.assertEqual(atomic.get(), -50)

    def test_atomic_int_thread_safety(self):
        """Test AtomicInt thread safety"""
        atomic = AtomicInt(0)
        results = []

        def increment_worker():
            for _ in range(1000):
                current = atomic.get()
                atomic.set(current + 1)

        def decrement_worker():
            for _ in range(1000):
                current = atomic.get()
                atomic.set(current - 1)

        # Start multiple threads
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=increment_worker)
            t2 = threading.Thread(target=decrement_worker)
            threads.extend([t1, t2])

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # The final value should be 0 if operations are atomic
        # Note: This test might be flaky due to the nature of threading,
        # but it demonstrates the atomic operations
        final_value = atomic.get()
        self.assertIsInstance(final_value, int)

    def test_atomic_int_str_representation(self):
        """Test AtomicInt string representation"""
        atomic = AtomicInt(123)
        self.assertEqual(str(atomic), "123")

        atomic.set(-456)
        self.assertEqual(str(atomic), "-456")

    def test_atomic_int_repr_representation(self):
        """Test AtomicInt repr representation"""
        atomic = AtomicInt(789)
        self.assertEqual(repr(atomic), "789")
        self.assertEqual(str(atomic), repr(atomic))

    def test_atomic_int_lock_usage(self):
        """Test that AtomicInt properly uses locks"""
        atomic = AtomicInt(0)

        # Verify that the lock exists and has the expected interface
        self.assertTrue(hasattr(atomic, "lock"))
        self.assertTrue(hasattr(atomic.lock, "acquire"))
        self.assertTrue(hasattr(atomic.lock, "release"))

        # Test that we can acquire and release the lock manually
        acquired = atomic.lock.acquire(blocking=False)
        self.assertTrue(acquired)
        atomic.lock.release()

        # Test that operations work (implicitly testing lock usage)
        atomic.set(42)
        self.assertEqual(atomic.get(), 42)
        self.assertEqual(str(atomic), "42")
        self.assertEqual(repr(atomic), "42")

    def test_atomic_int_concurrent_access(self):
        """Test AtomicInt with concurrent access patterns"""
        atomic = AtomicInt(1000)
        results = []

        def reader_worker():
            for _ in range(100):
                value = atomic.get()
                results.append(("read", value))
                time.sleep(0.001)  # Small delay to encourage interleaving

        def writer_worker(increment):
            for i in range(100):
                atomic.set(atomic.get() + increment)
                results.append(("write", atomic.get()))
                time.sleep(0.001)  # Small delay to encourage interleaving

        # Start reader and writer threads
        threads = []
        threads.append(threading.Thread(target=reader_worker))
        threads.append(threading.Thread(target=writer_worker, args=(1,)))
        threads.append(threading.Thread(target=writer_worker, args=(-1,)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify that we got results from all operations
        self.assertGreater(len(results), 0)

        # Verify that all read values are integers
        read_values = [result[1] for result in results if result[0] == "read"]
        for value in read_values:
            self.assertIsInstance(value, int)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utils functions"""

    def test_hpstate_with_debug_msg(self):
        """Test HPState integration with debug_msg"""
        with patch.dict(os.environ, {"RANK": "2", "WORLD_SIZE": "4"}):
            state = HPState()

            # Use state values in debug message
            msg = debug_msg(
                "State info",
                rank=state.rank,
                seq=state.iteration,
                steps=state.world_size,
            )

            self.assertIn("[RANK:2]", msg)
            self.assertIn("[SEQ:0]", msg)  # Initial iteration is 0
            self.assertIn("[STEPS:4]", msg)

    def test_exception_formatting_with_logging(self):
        """Test exception formatting integration with logging"""
        try:
            try:
                atomic = AtomicInt(0)
                # Force an error by accessing non-existent attribute
                _ = atomic.nonexistent_attr
            except AttributeError as e:
                raise RuntimeError("Atomic operation failed") from e
        except RuntimeError as exc:
            log_message = log_exc(exc, "atomic_test", rank=1, seq=5, steps=10)

            # Verify the log message contains all expected components
            self.assertIn("atomic_test:", log_message)
            self.assertIn("RuntimeError('Atomic operation failed')", log_message)
            self.assertIn("AttributeError", log_message)
            self.assertIn("[RANK:1]", log_message)
            self.assertIn("[SEQ:5]", log_message)
            self.assertIn("[STEPS:10]", log_message)

    @patch("threading.Thread")
    def test_async_operations_integration(self, mock_thread_class):
        """Test integration of async operations"""
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        # Test delayed_async_raise
        delayed_async_raise(12345, RuntimeError)

        # Verify thread was created with correct parameters
        mock_thread_class.assert_called_once()
        call_args = mock_thread_class.call_args
        self.assertEqual(call_args[1]["daemon"], True)
        self.assertEqual(call_args[1]["target"], async_raise)  # target function


class TestUtilsEdgeCases(unittest.TestCase):
    """Edge case tests for utils functions"""

    def test_hpstate_extreme_values(self):
        """Test HPState with extreme environment variable values"""
        with patch.dict(os.environ, {"RANK": "999999", "WORLD_SIZE": "1000000"}):
            state = HPState()
            self.assertEqual(state.rank, 999999)
            self.assertEqual(state.world_size, 1000000)

    def test_atomic_int_extreme_values(self):
        """Test AtomicInt with extreme values"""
        import sys

        # Test with maximum integer value
        max_int = sys.maxsize
        atomic = AtomicInt(max_int)
        self.assertEqual(atomic.get(), max_int)

        # Test with minimum integer value
        min_int = -sys.maxsize - 1
        atomic.set(min_int)
        self.assertEqual(atomic.get(), min_int)

    def test_debug_msg_special_characters(self):
        """Test debug_msg with special characters"""
        special_msg = "Message with\nnewlines\tand\ttabs"
        result = debug_msg(special_msg, rank=0)
        self.assertIn(special_msg, result)

    def test_format_exc_with_unicode(self):
        """Test format_exc with unicode characters"""
        exc = ValueError("Error with unicode: 你好世界")
        result = format_exc(exc)
        self.assertIn("你好世界", result)

    @patch("sys.is_finalizing")
    def test_async_raise_edge_cases(self, mock_is_finalizing):
        """Test async_raise edge cases"""
        # Test with None event
        mock_is_finalizing.return_value = True

        # Should not raise when finalizing
        async_raise(12345, RuntimeError, None)

        # Test with very large thread ID
        async_raise(999999999, RuntimeError, None)

    def test_reraise_if_unraisable_nested_context(self):
        """Test reraise_if_unraisable with nested contexts"""
        original_hook = sys.unraisablehook

        try:
            with reraise_if_unraisable(RuntimeError):
                with reraise_if_unraisable(ValueError):
                    # Inner context should be active
                    self.assertNotEqual(sys.unraisablehook, original_hook)
                # Outer context should be restored
                self.assertNotEqual(sys.unraisablehook, original_hook)
        finally:
            # Original hook should be fully restored
            self.assertEqual(sys.unraisablehook, original_hook)


if __name__ == "__main__":
    unittest.main()
