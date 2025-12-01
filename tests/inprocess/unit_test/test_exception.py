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

import inspect
import logging
import unittest
from unittest.mock import MagicMock, patch

from hyperpod_checkpointless_training.inprocess.exception import (
    HealthCheckError,
    InternalError,
    RankShouldRestart,
    RestartAbort,
    RestartError,
    TimeoutError,
)


class TestRankShouldRestart(unittest.TestCase):
    """Test the RankShouldRestart exception class"""

    def test_inheritance(self):
        """Test that RankShouldRestart inherits from BaseException"""
        self.assertTrue(issubclass(RankShouldRestart, BaseException))
        self.assertFalse(issubclass(RankShouldRestart, Exception))

    def test_instantiation(self):
        """Test that RankShouldRestart can be instantiated"""
        exc = RankShouldRestart()
        self.assertIsInstance(exc, RankShouldRestart)
        self.assertIsInstance(exc, BaseException)

    def test_instantiation_with_message(self):
        """Test RankShouldRestart with custom message"""
        message = "Custom restart message"
        exc = RankShouldRestart(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that RankShouldRestart can be raised and caught"""
        with self.assertRaises(RankShouldRestart):
            raise RankShouldRestart("Test restart")

    def test_can_be_caught_as_base_exception(self):
        """Test that RankShouldRestart can be caught as BaseException"""
        with self.assertRaises(BaseException):
            raise RankShouldRestart("Test restart")

    @patch('logging.getLogger')
    def test_del_method_with_debug_disabled(self, mock_get_logger):
        """Test __del__ method when debug logging is disabled"""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False
        mock_get_logger.return_value = mock_logger

        exc = RankShouldRestart()
        del exc

        mock_get_logger.assert_called_once_with('hyperpod_checkpointless_training.inprocess.exception')
        mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
        mock_logger.debug.assert_not_called()

    @patch('logging.getLogger')
    @patch('inspect.stack')
    def test_del_method_with_debug_enabled_short_stack(self, mock_stack, mock_get_logger):
        """Test __del__ method with debug enabled but short stack"""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = mock_logger

        # Mock a short stack (length <= 1)
        mock_stack.return_value = [MagicMock()]

        exc = RankShouldRestart()
        del exc

        mock_get_logger.assert_called_once_with("hyperpod_checkpointless_training.inprocess.exception")
        mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
        mock_stack.assert_called_once_with(context=0)
        mock_logger.debug.assert_not_called()

    @patch('logging.getLogger')
    @patch('inspect.stack')
    def test_del_method_with_debug_enabled_from_wrap_file(self, mock_stack, mock_get_logger):
        """Test __del__ method when called from wrap.py file"""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = mock_logger

        # Mock stack with wrap.py as the calling file
        from hyperpod_checkpointless_training.inprocess import wrap

        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        mock_frame1.filename = wrap.__file__
        mock_frame2.filename = "/some/other/file.py"

        mock_stack.return_value = [mock_frame1, mock_frame1, mock_frame2]

        exc = RankShouldRestart()
        del exc

        mock_get_logger.assert_called_once_with("hyperpod_checkpointless_training.inprocess.exception")
        mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
        mock_stack.assert_called_once_with(context=0)
        mock_logger.debug.assert_not_called()

    @patch('inspect.stack')
    def test_del_method_with_debug_enabled_from_other_file(self, mock_stack):
        """Test __del__ method when called from non-wrap file"""
        # Mock stack frames
        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        mock_frame3 = MagicMock()

        mock_frame1.filename = "/current/file.py"
        mock_frame2.filename = "/caller/file.py"
        mock_frame3.filename = "/other/file.py"

        mock_frame2.frame.f_code.co_filename = "/caller/file.py"
        mock_frame2.frame.f_lineno = 42
        mock_frame3.frame.f_code.co_filename = "/other/file.py"
        mock_frame3.frame.f_lineno = 123

        mock_stack.return_value = [mock_frame1, mock_frame2, mock_frame3]

        # Patch the specific logger for this module
        with patch("hyperpod_checkpointless_training.inprocess.exception.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.isEnabledFor.return_value = True
            mock_get_logger.return_value = mock_logger

            exc = RankShouldRestart()
            del exc

            # Check that getLogger was called with our module name (among other calls)
            mock_get_logger.assert_any_call("hyperpod_checkpointless_training.inprocess.exception")
            mock_logger.isEnabledFor.assert_called_with(logging.DEBUG)
            mock_stack.assert_called_once_with(context=0)

            # Should log the traceback
            expected_traceback = "/caller/file.py:42 <- /other/file.py:123"
            mock_logger.debug.assert_called_with(f"RankShouldRestart suppressed at {expected_traceback}")


class TestRestartError(unittest.TestCase):
    """Test the RestartError exception class"""

    def test_inheritance(self):
        """Test that RestartError inherits from Exception"""
        self.assertTrue(issubclass(RestartError, Exception))
        self.assertTrue(issubclass(RestartError, BaseException))

    def test_instantiation(self):
        """Test that RestartError can be instantiated"""
        exc = RestartError()
        self.assertIsInstance(exc, RestartError)
        self.assertIsInstance(exc, Exception)

    def test_instantiation_with_message(self):
        """Test RestartError with custom message"""
        message = "Custom restart error message"
        exc = RestartError(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that RestartError can be raised and caught"""
        with self.assertRaises(RestartError):
            raise RestartError("Test error")

    def test_can_be_caught_as_exception(self):
        """Test that RestartError can be caught as Exception"""
        with self.assertRaises(Exception):
            raise RestartError("Test error")


class TestRestartAbort(unittest.TestCase):
    """Test the RestartAbort exception class"""

    def test_inheritance(self):
        """Test that RestartAbort inherits from BaseException"""
        self.assertTrue(issubclass(RestartAbort, BaseException))
        self.assertFalse(issubclass(RestartAbort, Exception))

    def test_instantiation(self):
        """Test that RestartAbort can be instantiated"""
        exc = RestartAbort()
        self.assertIsInstance(exc, RestartAbort)
        self.assertIsInstance(exc, BaseException)

    def test_instantiation_with_message(self):
        """Test RestartAbort with custom message"""
        message = "Custom abort message"
        exc = RestartAbort(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that RestartAbort can be raised and caught"""
        with self.assertRaises(RestartAbort):
            raise RestartAbort("Test abort")

    def test_can_be_caught_as_base_exception(self):
        """Test that RestartAbort can be caught as BaseException"""
        with self.assertRaises(BaseException):
            raise RestartAbort("Test abort")


class TestHealthCheckError(unittest.TestCase):
    """Test the HealthCheckError exception class"""

    def test_inheritance(self):
        """Test that HealthCheckError inherits from RestartError"""
        self.assertTrue(issubclass(HealthCheckError, RestartError))
        self.assertTrue(issubclass(HealthCheckError, Exception))
        self.assertTrue(issubclass(HealthCheckError, BaseException))

    def test_instantiation(self):
        """Test that HealthCheckError can be instantiated"""
        exc = HealthCheckError()
        self.assertIsInstance(exc, HealthCheckError)
        self.assertIsInstance(exc, RestartError)
        self.assertIsInstance(exc, Exception)

    def test_instantiation_with_message(self):
        """Test HealthCheckError with custom message"""
        message = "Health check failed"
        exc = HealthCheckError(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that HealthCheckError can be raised and caught"""
        with self.assertRaises(HealthCheckError):
            raise HealthCheckError("Health check failed")

    def test_can_be_caught_as_restart_error(self):
        """Test that HealthCheckError can be caught as RestartError"""
        with self.assertRaises(RestartError):
            raise HealthCheckError("Health check failed")

    def test_can_be_caught_as_exception(self):
        """Test that HealthCheckError can be caught as Exception"""
        with self.assertRaises(Exception):
            raise HealthCheckError("Health check failed")


class TestInternalError(unittest.TestCase):
    """Test the InternalError exception class"""

    def test_inheritance(self):
        """Test that InternalError inherits from RestartError"""
        self.assertTrue(issubclass(InternalError, RestartError))
        self.assertTrue(issubclass(InternalError, Exception))
        self.assertTrue(issubclass(InternalError, BaseException))

    def test_instantiation(self):
        """Test that InternalError can be instantiated"""
        exc = InternalError()
        self.assertIsInstance(exc, InternalError)
        self.assertIsInstance(exc, RestartError)
        self.assertIsInstance(exc, Exception)

    def test_instantiation_with_message(self):
        """Test InternalError with custom message"""
        message = "Internal error occurred"
        exc = InternalError(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that InternalError can be raised and caught"""
        with self.assertRaises(InternalError):
            raise InternalError("Internal error")

    def test_can_be_caught_as_restart_error(self):
        """Test that InternalError can be caught as RestartError"""
        with self.assertRaises(RestartError):
            raise InternalError("Internal error")

    def test_can_be_caught_as_exception(self):
        """Test that InternalError can be caught as Exception"""
        with self.assertRaises(Exception):
            raise InternalError("Internal error")


class TestTimeoutError(unittest.TestCase):
    """Test the TimeoutError exception class"""

    def test_inheritance(self):
        """Test that TimeoutError inherits from RestartError"""
        self.assertTrue(issubclass(TimeoutError, RestartError))
        self.assertTrue(issubclass(TimeoutError, Exception))
        self.assertTrue(issubclass(TimeoutError, BaseException))

    def test_instantiation(self):
        """Test that TimeoutError can be instantiated"""
        exc = TimeoutError()
        self.assertIsInstance(exc, TimeoutError)
        self.assertIsInstance(exc, RestartError)
        self.assertIsInstance(exc, Exception)

    def test_instantiation_with_message(self):
        """Test TimeoutError with custom message"""
        message = "Operation timed out"
        exc = TimeoutError(message)
        self.assertEqual(str(exc), message)

    def test_can_be_raised(self):
        """Test that TimeoutError can be raised and caught"""
        with self.assertRaises(TimeoutError):
            raise TimeoutError("Timeout occurred")

    def test_can_be_caught_as_restart_error(self):
        """Test that TimeoutError can be caught as RestartError"""
        with self.assertRaises(RestartError):
            raise TimeoutError("Timeout occurred")

    def test_can_be_caught_as_exception(self):
        """Test that TimeoutError can be caught as Exception"""
        with self.assertRaises(Exception):
            raise TimeoutError("Timeout occurred")


class TestExceptionHierarchy(unittest.TestCase):
    """Test the exception hierarchy and relationships"""

    def test_base_exception_hierarchy(self):
        """Test BaseException-derived exceptions"""
        base_exceptions = [RankShouldRestart, RestartAbort]
        
        for exc_class in base_exceptions:
            with self.subTest(exception=exc_class.__name__):
                self.assertTrue(issubclass(exc_class, BaseException))
                self.assertFalse(issubclass(exc_class, Exception))

    def test_exception_hierarchy(self):
        """Test Exception-derived exceptions"""
        exception_classes = [RestartError, HealthCheckError, InternalError, TimeoutError]
        
        for exc_class in exception_classes:
            with self.subTest(exception=exc_class.__name__):
                self.assertTrue(issubclass(exc_class, Exception))
                self.assertTrue(issubclass(exc_class, BaseException))

    def test_restart_error_subclasses(self):
        """Test that all RestartError subclasses inherit correctly"""
        restart_error_subclasses = [HealthCheckError, InternalError, TimeoutError]
        
        for exc_class in restart_error_subclasses:
            with self.subTest(exception=exc_class.__name__):
                self.assertTrue(issubclass(exc_class, RestartError))

    def test_exception_catching_hierarchy(self):
        """Test exception catching behavior"""
        # Test that RestartError subclasses can be caught by RestartError
        restart_error_subclasses = [HealthCheckError, InternalError, TimeoutError]
        
        for exc_class in restart_error_subclasses:
            with self.subTest(exception=exc_class.__name__):
                try:
                    raise exc_class("Test message")
                except RestartError:
                    pass  # Should be caught
                except Exception:
                    self.fail(f"{exc_class.__name__} should be caught by RestartError")

    def test_base_exception_not_caught_by_exception(self):
        """Test that BaseException subclasses are not caught by Exception"""
        base_exceptions = [RankShouldRestart, RestartAbort]
        
        for exc_class in base_exceptions:
            with self.subTest(exception=exc_class.__name__):
                caught_by_exception = False
                try:
                    try:
                        raise exc_class("Test message")
                    except Exception:
                        caught_by_exception = True
                except BaseException:
                    pass  # Expected to reach here
                
                self.assertFalse(caught_by_exception, 
                               f"{exc_class.__name__} should not be caught by Exception")


class TestExceptionIntegration(unittest.TestCase):
    """Integration tests for exception behavior"""

    def test_exception_chaining(self):
        """Test exception chaining with custom exceptions"""
        original_error = ValueError("Original error")
        
        try:
            try:
                raise original_error
            except ValueError as e:
                raise InternalError("Wrapper error") from e
        except InternalError as e:
            self.assertEqual(str(e), "Wrapper error")
            self.assertIs(e.__cause__, original_error)

    def test_exception_with_args(self):
        """Test exceptions with multiple arguments"""
        args = ("Error message", 42, {"key": "value"})
        
        exc = RestartError(*args)
        self.assertEqual(exc.args, args)

    def test_exception_repr(self):
        """Test string representation of exceptions"""
        message = "Test error message"
        
        exceptions = [
            RankShouldRestart(message),
            RestartError(message),
            RestartAbort(message),
            HealthCheckError(message),
            InternalError(message),
            TimeoutError(message)
        ]
        
        for exc in exceptions:
            with self.subTest(exception=type(exc).__name__):
                self.assertEqual(str(exc), message)
                self.assertIn(type(exc).__name__, repr(exc))
                self.assertIn(message, repr(exc))

    def test_exception_equality(self):
        """Test exception equality comparison"""
        msg = "Test message"
        
        # Same type, same message
        exc1 = RestartError(msg)
        exc2 = RestartError(msg)
        self.assertEqual(exc1.args, exc2.args)
        
        # Different types, same message
        exc3 = InternalError(msg)
        self.assertNotEqual(type(exc1), type(exc3))
        self.assertEqual(exc1.args, exc3.args)

    def test_exception_in_finally_block(self):
        """Test exception behavior in finally blocks"""
        cleanup_called = False
        
        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        try:
            try:
                raise HealthCheckError("Health check failed")
            finally:
                cleanup()
        except HealthCheckError:
            pass
        
        self.assertTrue(cleanup_called)

    def test_multiple_exception_types_in_except_clause(self):
        """Test catching multiple exception types"""
        exceptions_to_test = [
            HealthCheckError("Health check failed"),
            InternalError("Internal error"),
            TimeoutError("Timeout occurred")
        ]
        
        for exc in exceptions_to_test:
            with self.subTest(exception=type(exc).__name__):
                try:
                    raise exc
                except (HealthCheckError, InternalError, TimeoutError) as caught:
                    self.assertIs(caught, exc)


if __name__ == "__main__":
    unittest.main()
