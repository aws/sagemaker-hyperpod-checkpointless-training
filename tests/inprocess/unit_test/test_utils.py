import pytest
import os
import threading
import socket
import sys
from unittest.mock import Mock, patch, MagicMock
import ctypes

from hyperpod_checkpointless_training.inprocess.utils import (
    async_raise,
    delayed_async_raise,
    reraise_if_unraisable,
    HPState,
    debug_msg,
    format_exc,
    log_exc,
    AtomicInt,
)


class TestAsyncRaise:
    """Test async_raise function"""

    @patch('ctypes.pythonapi.PyThreadState_SetAsyncExc')
    def test_async_raise_success(self, mock_set_async_exc):
        """Test successful async_raise call"""
        mock_set_async_exc.return_value = 1
        
        # Should not raise exception
        async_raise(123, ValueError)
        
        mock_set_async_exc.assert_called_once_with(123, ValueError)

    @patch('ctypes.pythonapi.PyThreadState_SetAsyncExc')
    def test_async_raise_with_event(self, mock_set_async_exc):
        """Test async_raise with event"""
        mock_set_async_exc.return_value = 1
        event = threading.Event()
        event.set()
        
        async_raise(123, ValueError, event)
        
        mock_set_async_exc.assert_called_once_with(123, ValueError)

    @patch('ctypes.pythonapi.PyThreadState_SetAsyncExc')
    def test_async_raise_thread_not_found(self, mock_set_async_exc):
        """Test async_raise when thread not found"""
        mock_set_async_exc.return_value = 0
        
        with pytest.raises(RuntimeError):
            async_raise(123, ValueError)

    @patch('ctypes.pythonapi.PyThreadState_SetAsyncExc')
    def test_async_raise_multiple_threads(self, mock_set_async_exc):
        """Test async_raise when multiple threads affected"""
        mock_set_async_exc.side_effect = [2, 1]  # First call returns 2, second returns 1
        
        with pytest.raises(RuntimeError):
            async_raise(123, ValueError)
        
        # Should call twice - once with exception, once with None to clean up
        assert mock_set_async_exc.call_count == 2
        mock_set_async_exc.assert_any_call(123, ValueError)
        mock_set_async_exc.assert_any_call(123, None)

    @patch('sys.is_finalizing')
    @patch('ctypes.pythonapi.PyThreadState_SetAsyncExc')
    def test_async_raise_during_finalization(self, mock_set_async_exc, mock_is_finalizing):
        """Test async_raise during interpreter finalization"""
        mock_is_finalizing.return_value = True
        
        # Should not call PyThreadState_SetAsyncExc during finalization
        async_raise(123, ValueError)
        
        mock_set_async_exc.assert_not_called()


class TestDelayedAsyncRaise:
    """Test delayed_async_raise function"""

    @patch("hyperpod_checkpointless_training.inprocess.utils.async_raise")
    @patch("threading.Thread")
    def test_delayed_async_raise(self, mock_thread_class, mock_async_raise):
        """Test delayed_async_raise creates and starts thread"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        delayed_async_raise(123, ValueError)

        # Verify thread was created with correct arguments
        mock_thread_class.assert_called_once()
        args, kwargs = mock_thread_class.call_args
        assert kwargs['daemon'] is True
        assert kwargs['target'] == mock_async_raise

        # Verify thread was started
        mock_thread.start.assert_called_once()


class TestReraiseIfUnraisable:
    """Test reraise_if_unraisable context manager"""

    def test_reraise_if_unraisable_context_manager(self):
        """Test reraise_if_unraisable as context manager"""
        original_hook = sys.unraisablehook
        
        with reraise_if_unraisable(ValueError):
            # unraisablehook should be wrapped
            assert sys.unraisablehook != original_hook
        
        # Should restore original hook
        assert sys.unraisablehook == original_hook

class TestHPState:
    """Test HPState class"""

    def test_hpstate_initialization(self):
        """Test HPState initialization"""
        with patch.dict(os.environ, {'RANK': '2', 'WORLD_SIZE': '4'}):
            state = HPState()
            
            assert state.rank == 2
            assert state.world_size == 4
            assert state.iteration == 0

    def test_hpstate_initialization_defaults(self):
        """Test HPState initialization with default values"""
        with patch.dict(os.environ, {}, clear=True):
            state = HPState()
            
            assert state.rank == -1
            assert state.world_size == -1
            assert state.iteration == 0

    def test_hpstate_advance(self):
        """Test HPState advance method"""
        state = HPState()
        initial_iteration = state.iteration
        
        state.advance()
        
        assert state.iteration == initial_iteration + 1

    def test_hpstate_set_distributed_vars_raises(self):
        """Test HPState set_distributed_vars raises RuntimeError"""
        state = HPState()
        
        with pytest.raises(RuntimeError, match="HPState state does not allow to set dist vars"):
            state.set_distributed_vars()

    def test_hpstate_get_distributed_vars(self):
        """Test HPState get_distributed_vars method"""
        with patch.dict(os.environ, {'RANK': '3', 'WORLD_SIZE': '8'}):
            state = HPState()
            state.rank = 999  # Set to different value
            state.world_size = 999
            
            state.get_distributed_vars()
            
            assert state.rank == 3
            assert state.world_size == 8


class TestDebugMsg:
    """Test debug_msg function"""

    @patch('socket.gethostname')
    @patch('threading.get_ident')
    @patch('os.getpid')
    def test_debug_msg_with_defaults(self, mock_getpid, mock_get_ident, mock_gethostname):
        """Test debug_msg with default parameters"""
        mock_getpid.return_value = 12345
        mock_get_ident.return_value = 67890
        mock_gethostname.return_value = 'test-host'
        
        with patch.dict(os.environ, {'RANK': '2', 'JOB_RESTART_COUNT': '3', 'SPARE': 'true'}):
            result = debug_msg("test message")
            
            expected = "[RANK:2][SPARE:true][SEQ:3][STEPS:-1][SPARE:true][PID:12345][TID:67890][HOST:test-host] test message"
            assert result == expected

    def test_debug_msg_with_custom_params(self):
        """Test debug_msg with custom parameters"""
        with patch('socket.gethostname', return_value='custom-host'), \
             patch('threading.get_ident', return_value=11111), \
             patch('os.getpid', return_value=22222):
            
            result = debug_msg("custom message", rank=5, seq=7, steps=100)
            
            expected = "[RANK:5][SPARE:None][SEQ:7][STEPS:100][SPARE:None][PID:22222][TID:11111][HOST:custom-host] custom message"
            assert result == expected

    def test_debug_msg_missing_env_vars(self):
        """Test debug_msg with missing environment variables"""
        with patch.dict(os.environ, {}, clear=True), \
             patch('socket.gethostname', return_value='test-host'), \
             patch('threading.get_ident', return_value=11111), \
             patch('os.getpid', return_value=22222):
            
            result = debug_msg("test message")
            
            expected = "[RANK:-1][SPARE:None][SEQ:-1][STEPS:-1][SPARE:None][PID:22222][TID:11111][HOST:test-host] test message"
            assert result == expected


class TestFormatExc:
    """Test format_exc function"""

    def test_format_exc_single_exception(self):
        """Test format_exc with single exception"""
        exc = ValueError("test error")
        result = format_exc(exc)
        
        assert result == "ValueError('test error')"

    def test_format_exc_chained_exceptions(self):
        """Test format_exc with chained exceptions"""
        try:
            try:
                raise ValueError("original error")
            except ValueError as e:
                raise RuntimeError("wrapper error") from e
        except RuntimeError as exc:
            result = format_exc(exc)
            
            assert "RuntimeError('wrapper error')" in result
            assert "ValueError('original error')" in result
            assert " <- " in result

    def test_format_exc_no_cause(self):
        """Test format_exc with exception that has no cause"""
        exc = TypeError("simple error")
        result = format_exc(exc)
        
        assert result == "TypeError('simple error')"


class TestLogExc:
    """Test log_exc function"""

    def test_log_exc_basic(self):
        """Test log_exc basic functionality"""
        exc = ValueError("test error")
        result = log_exc(exc, "test_function")
        
        assert "test_function:" in result
        assert "ValueError('test error')" in result
        assert "[RANK:-1]" in result

    def test_log_exc_with_params(self):
        """Test log_exc with custom parameters"""
        exc = RuntimeError("runtime error")
        result = log_exc(exc, "custom_function", rank=3, seq=5, steps=10)
        
        assert "custom_function:" in result
        assert "RuntimeError('runtime error')" in result
        assert "[RANK:3]" in result
        assert "[SEQ:5]" in result
        assert "[STEPS:10]" in result


class TestAtomicInt:
    """Test AtomicInt class"""

    def test_atomic_int_initialization(self):
        """Test AtomicInt initialization"""
        atomic = AtomicInt(42)
        assert atomic.get() == 42

    def test_atomic_int_set_get(self):
        """Test AtomicInt set and get operations"""
        atomic = AtomicInt(0)
        
        atomic.set(100)
        assert atomic.get() == 100
        
        atomic.set(-50)
        assert atomic.get() == -50

    def test_atomic_int_thread_safety(self):
        """Test AtomicInt thread safety"""
        atomic = AtomicInt(0)
        results = []
        
        def increment():
            for _ in range(100):
                current = atomic.get()
                atomic.set(current + 1)
        
        def decrement():
            for _ in range(100):
                current = atomic.get()
                atomic.set(current - 1)
        
        # Run concurrent operations
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=increment)
            t2 = threading.Thread(target=decrement)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Final value should be 0 (500 increments - 500 decrements)
        assert atomic.get() == 0

    def test_atomic_int_str_repr(self):
        """Test AtomicInt string representation"""
        atomic = AtomicInt(123)
        
        assert str(atomic) == "123"
        assert repr(atomic) == "123"
        
        atomic.set(-456)
        assert str(atomic) == "-456"
        assert repr(atomic) == "-456"
