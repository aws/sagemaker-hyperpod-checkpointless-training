import datetime
import threading
import time
import unittest
import warnings
from unittest.mock import MagicMock, call, patch

from hyperpod_checkpointless_training.inprocess.exception import TimeoutError
from hyperpod_checkpointless_training.inprocess.finalize import Finalize, ThreadedFinalize
from hyperpod_checkpointless_training.inprocess.utils import HPState


class TestFinalizeAbstractBase(unittest.TestCase):
    """Test the abstract base Finalize class"""

    def test_finalize_is_abstract(self):
        """Test that Finalize cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            Finalize()

    def test_finalize_abstract_method(self):
        """Test that __call__ is an abstract method"""

        class IncompleteFinalize(Finalize):
            pass

        with self.assertRaises(TypeError):
            IncompleteFinalize()

    def test_finalize_concrete_implementation(self):
        """Test that a concrete implementation can be created"""

        class ConcreteFinalize(Finalize):
            def __call__(self, state, train_ex=None):
                return state, train_ex

        finalize = ConcreteFinalize()
        state = HPState()
        result_state, result_ex = finalize(state)

        self.assertEqual(result_state, state)
        self.assertIsNone(result_ex)

    def test_finalize_concrete_implementation_with_exception(self):
        """Test concrete implementation with exception"""

        class ConcreteFinalize(Finalize):
            def __call__(self, state, train_ex=None):
                return state, train_ex

        finalize = ConcreteFinalize()
        state = HPState()
        exception = ValueError("test error")
        result_state, result_ex = finalize(state, exception)

        self.assertEqual(result_state, state)
        self.assertEqual(result_ex, exception)


class TestThreadedFinalize(unittest.TestCase):
    """Test the ThreadedFinalize class"""

    def setUp(self):
        self.state = HPState()
        self.timeout = datetime.timedelta(seconds=1)

    def test_init_with_defaults(self):
        """Test ThreadedFinalize initialization with default arguments"""
        mock_fn = MagicMock()
        finalize = ThreadedFinalize(self.timeout, mock_fn)

        self.assertEqual(finalize.timeout, self.timeout)
        self.assertEqual(finalize.fn, mock_fn)
        self.assertEqual(finalize.args, ())
        self.assertEqual(finalize.kwargs, {})

    def test_init_with_args_and_kwargs(self):
        """Test ThreadedFinalize initialization with custom args and kwargs"""
        mock_fn = MagicMock()
        args = (1, 2, 3)
        kwargs = {"key": "value", "number": 42}

        finalize = ThreadedFinalize(self.timeout, mock_fn, args, kwargs)

        self.assertEqual(finalize.timeout, self.timeout)
        self.assertEqual(finalize.fn, mock_fn)
        self.assertEqual(finalize.args, args)
        self.assertEqual(finalize.kwargs, kwargs)

    def test_init_with_none_kwargs(self):
        """Test ThreadedFinalize initialization with None kwargs"""
        mock_fn = MagicMock()
        finalize = ThreadedFinalize(self.timeout, mock_fn, kwargs=None)

        self.assertEqual(finalize.kwargs, {})

    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_successful_execution(self, mock_getenv, mock_thread_class):
        """Test successful execution of finalize function"""
        mock_getenv.return_value = "0"
        mock_fn = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        finalize = ThreadedFinalize(self.timeout, mock_fn)
        result = finalize(self.state)

        # Verify thread creation
        mock_thread_class.assert_called_once_with(
            target=mock_fn, name="ThreadedFinalize-0", args=(), kwargs={}, daemon=True
        )

        # Verify thread execution
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once_with(self.timeout.total_seconds())

        # Verify return value (should be None for successful execution)
        self.assertIsNone(result)

    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_with_custom_args_kwargs(self, mock_getenv, mock_thread_class):
        """Test execution with custom args and kwargs"""
        mock_getenv.return_value = "1"
        mock_fn = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        args = ("arg1", "arg2")
        kwargs = {"param": "value"}
        finalize = ThreadedFinalize(self.timeout, mock_fn, args, kwargs)

        result = finalize(self.state)

        # Verify thread creation with custom args/kwargs
        mock_thread_class.assert_called_once_with(
            target=mock_fn,
            name="ThreadedFinalize-1",
            args=args,
            kwargs=kwargs,
            daemon=True,
        )

        # Verify return value (should be None for successful execution)
        self.assertIsNone(result)

    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_with_train_exception(self, mock_getenv, mock_thread_class):
        """Test execution when train_ex is provided"""
        mock_getenv.return_value = "0"
        mock_fn = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread

        finalize = ThreadedFinalize(self.timeout, mock_fn)
        train_exception = ValueError("Training error")

        result = finalize(self.state, train_exception)

        # The method doesn't return the exception, it just completes successfully
        self.assertIsNone(result)

    @patch("threading.Thread")
    @patch("os.getenv")
    def test_call_timeout_raises_exception(self, mock_getenv, mock_thread_class):
        """Test that timeout raises TimeoutError"""
        mock_getenv.return_value = "0"
        mock_fn = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Thread still alive after timeout
        mock_thread_class.return_value = mock_thread

        finalize = ThreadedFinalize(self.timeout, mock_fn)

        with self.assertRaises(TimeoutError):
            finalize(self.state)

        # Verify thread was started and joined
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once_with(self.timeout.total_seconds())
        mock_thread.is_alive.assert_called_once()

    @patch("os.getenv")
    def test_call_different_rank_values(self, mock_getenv):
        """Test thread naming with different RANK values"""
        test_ranks = ["0", "1", "5", "10"]

        for rank in test_ranks:
            with self.subTest(rank=rank):
                mock_getenv.return_value = rank
                mock_fn = MagicMock()

                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = MagicMock()
                    mock_thread.is_alive.return_value = False
                    mock_thread_class.return_value = mock_thread

                    finalize = ThreadedFinalize(self.timeout, mock_fn)
                    finalize(self.state)

                    expected_name = f"ThreadedFinalize-{rank}"
                    mock_thread_class.assert_called_once()
                    call_kwargs = mock_thread_class.call_args[1]
                    self.assertEqual(call_kwargs["name"], expected_name)

    @patch("os.getenv")
    def test_call_rank_default_value(self, mock_getenv):
        """Test thread naming when RANK environment variable is not set"""
        mock_getenv.return_value = "0"  # Return string "0" as default
        mock_fn = MagicMock()

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            finalize = ThreadedFinalize(self.timeout, mock_fn)
            finalize(self.state)

            # When RANK is not set, os.getenv should return the default value 0
            mock_getenv.assert_called_once_with("RANK", 0)

    def test_timeout_conversion(self):
        """Test that timeout is properly converted to seconds"""
        mock_fn = MagicMock()
        timeout_delta = datetime.timedelta(minutes=2, seconds=30)  # 150 seconds

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            finalize = ThreadedFinalize(timeout_delta, mock_fn)
            finalize(self.state)

            mock_thread.join.assert_called_once_with(150.0)

    def test_thread_daemon_property(self):
        """Test that created thread is marked as daemon"""
        mock_fn = MagicMock()

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            finalize = ThreadedFinalize(self.timeout, mock_fn)
            finalize(self.state)

            # Verify daemon=True was passed
            call_kwargs = mock_thread_class.call_args[1]
            self.assertTrue(call_kwargs["daemon"])

    def test_function_execution_in_thread(self):
        """Test that the function is actually executed in the thread"""
        executed = threading.Event()

        def test_function():
            executed.set()

        finalize = ThreadedFinalize(datetime.timedelta(seconds=2), test_function)
        finalize(self.state)

        # Give some time for the thread to execute
        self.assertTrue(executed.wait(timeout=1))

    def test_function_with_arguments_execution(self):
        """Test that function is called with correct arguments"""
        result_container = {}

        def test_function(arg1, arg2, kwarg1=None, kwarg2=None):
            result_container["args"] = (arg1, arg2)
            result_container["kwargs"] = {"kwarg1": kwarg1, "kwarg2": kwarg2}

        args = ("test_arg1", "test_arg2")
        kwargs = {"kwarg1": "test_kwarg1", "kwarg2": "test_kwarg2"}

        finalize = ThreadedFinalize(
            datetime.timedelta(seconds=2), test_function, args, kwargs
        )
        finalize(self.state)

        # Give some time for the thread to execute
        time.sleep(0.1)

        self.assertEqual(result_container["args"], args)
        self.assertEqual(result_container["kwargs"], kwargs)

    def test_real_timeout_scenario(self):
        """Test actual timeout scenario with a slow function"""

        def slow_function():
            time.sleep(2)  # Sleep longer than timeout

        short_timeout = datetime.timedelta(milliseconds=100)
        finalize = ThreadedFinalize(short_timeout, slow_function)

        with self.assertRaises(TimeoutError):
            finalize(self.state)

    def test_successful_completion(self):
        """Test that successful execution returns None"""
        mock_fn = MagicMock()
        finalize = ThreadedFinalize(self.timeout, mock_fn)

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            result = finalize(self.state)

            # Successful execution should return None
            self.assertIsNone(result)

    def test_with_train_exception_parameter(self):
        """Test that train_ex parameter doesn't affect execution"""
        mock_fn = MagicMock()
        finalize = ThreadedFinalize(self.timeout, mock_fn)

        original_exception = RuntimeError("Original error")

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            result = finalize(self.state, original_exception)

            # The train_ex parameter doesn't affect the return value
            self.assertIsNone(result)


class TestThreadedFinalizeIntegration(unittest.TestCase):
    """Integration tests for ThreadedFinalize"""

    def setUp(self):
        self.state = HPState()

    def test_integration_successful_cleanup(self):
        """Integration test for successful cleanup operation"""
        cleanup_performed = threading.Event()
        cleanup_data = {}

        def cleanup_function(resource_id, cleanup_type="standard"):
            cleanup_data["resource_id"] = resource_id
            cleanup_data["cleanup_type"] = cleanup_type
            cleanup_performed.set()

        args = ("resource_123",)
        kwargs = {"cleanup_type": "thorough"}
        timeout = datetime.timedelta(seconds=1)

        finalize = ThreadedFinalize(timeout, cleanup_function, args, kwargs)
        result = finalize(self.state)

        # Wait for cleanup to complete
        self.assertTrue(cleanup_performed.wait(timeout=2))

        # Verify cleanup was performed correctly
        self.assertEqual(cleanup_data["resource_id"], "resource_123")
        self.assertEqual(cleanup_data["cleanup_type"], "thorough")

        # Verify return value
        self.assertIsNone(result)

    def test_integration_with_exception_context(self):
        """Integration test with exception context"""

        def dummy_cleanup():
            pass

        timeout = datetime.timedelta(seconds=1)
        finalize = ThreadedFinalize(timeout, dummy_cleanup)

        original_exception = ValueError("Training failed")
        result = finalize(self.state, original_exception)

        # The exception parameter doesn't affect the return value
        self.assertIsNone(result)


class TestThreadedFinalizeEdgeCases(unittest.TestCase):
    """Edge case tests for ThreadedFinalize"""

    def setUp(self):
        self.state = HPState()

    def test_zero_timeout(self):
        """Test with zero timeout"""

        def instant_function():
            pass

        zero_timeout = datetime.timedelta(seconds=0)
        finalize = ThreadedFinalize(zero_timeout, instant_function)

        # Should complete without timeout error for instant function
        result = finalize(self.state)
        self.assertIsNone(result)

    def test_very_short_timeout(self):
        """Test with very short timeout"""

        def quick_function():
            pass

        short_timeout = datetime.timedelta(microseconds=1)
        finalize = ThreadedFinalize(short_timeout, quick_function)

        # May or may not timeout depending on system performance
        try:
            result = finalize(self.state)
            self.assertIsNone(result)
        except TimeoutError:
            # This is also acceptable for very short timeouts
            pass

    def test_function_with_exception(self):
        """Test that exceptions in the target function don't propagate"""

        def failing_function():
            raise RuntimeError("Function failed")

        timeout = datetime.timedelta(seconds=1)
        finalize = ThreadedFinalize(timeout, failing_function)

        # The exception in the thread should not propagate to the main thread
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = finalize(self.state)
            self.assertIsNone(result)

    def test_multiple_calls_same_instance(self):
        """Test calling the same ThreadedFinalize instance multiple times"""
        call_count = 0

        def counting_function():
            nonlocal call_count
            call_count += 1

        timeout = datetime.timedelta(seconds=1)
        finalize = ThreadedFinalize(timeout, counting_function)

        # Call multiple times
        for i in range(3):
            result = finalize(self.state)
            self.assertIsNone(result)

        # Give threads time to complete
        time.sleep(0.1)
        self.assertEqual(call_count, 3)

    @patch("os.getenv")
    def test_rank_with_non_numeric_value(self, mock_getenv):
        """Test behavior when RANK environment variable is non-numeric"""
        mock_getenv.return_value = "invalid"
        mock_fn = MagicMock()

        finalize = ThreadedFinalize(datetime.timedelta(seconds=1), mock_fn)

        # Should raise ValueError when trying to convert "invalid" to int
        with self.assertRaises(ValueError):
            finalize(self.state)

    def test_large_timeout_value(self):
        """Test with very large timeout value"""

        def quick_function():
            pass

        large_timeout = datetime.timedelta(days=365)  # 1 year timeout
        finalize = ThreadedFinalize(large_timeout, quick_function)

        result = finalize(self.state)
        self.assertIsNone(result)

    def test_function_modifying_shared_state(self):
        """Test function that modifies shared state"""
        shared_data = {"modified": False}

        def modifying_function():
            shared_data["modified"] = True

        timeout = datetime.timedelta(seconds=1)
        finalize = ThreadedFinalize(timeout, modifying_function)

        result = finalize(self.state)

        # Give thread time to complete
        time.sleep(0.1)

        self.assertIsNone(result)
        self.assertTrue(shared_data["modified"])

    def test_thread_name_format(self):
        """Test that thread names are formatted correctly"""
        mock_fn = MagicMock()

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_class.return_value = mock_thread

            with patch("os.getenv", return_value="42"):
                finalize = ThreadedFinalize(datetime.timedelta(seconds=1), mock_fn)
                finalize(self.state)

                # Verify thread name includes class name and rank
                call_kwargs = mock_thread_class.call_args[1]
                self.assertEqual(call_kwargs["name"], "ThreadedFinalize-42")

    def test_datetime_timedelta_properties(self):
        """Test various datetime.timedelta properties"""
        test_cases = [
            datetime.timedelta(milliseconds=500),
            datetime.timedelta(seconds=1.5),
            datetime.timedelta(minutes=1),
            datetime.timedelta(hours=1),
        ]

        for timeout in test_cases:
            with self.subTest(timeout=timeout):
                mock_fn = MagicMock()
                finalize = ThreadedFinalize(timeout, mock_fn)

                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = MagicMock()
                    mock_thread.is_alive.return_value = False
                    mock_thread_class.return_value = mock_thread

                    finalize(self.state)

                    # Verify timeout conversion
                    expected_seconds = timeout.total_seconds()
                    mock_thread.join.assert_called_once_with(expected_seconds)


if __name__ == "__main__":
    unittest.main()
