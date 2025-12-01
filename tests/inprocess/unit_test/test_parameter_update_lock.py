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

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock


class TestParameterUpdateLock(unittest.TestCase):
    """Test the ParameterUpdateLock singleton class"""

    def setUp(self):
        """Reset singleton instance before each test"""
        # Clear singleton instance to ensure clean state for each test
        ParameterUpdateLock._instance = None

    def tearDown(self):
        """Clean up after each test"""
        # Clear singleton instance after each test
        ParameterUpdateLock._instance = None

    def test_singleton_behavior(self):
        """Test that ParameterUpdateLock is a singleton"""
        lock1 = ParameterUpdateLock()
        lock2 = ParameterUpdateLock()

        self.assertIs(lock1, lock2)
        self.assertEqual(id(lock1), id(lock2))

    def test_singleton_thread_safety(self):
        """Test that singleton creation is thread-safe"""
        instances = []
        barrier = threading.Barrier(5)

        def create_instance():
            barrier.wait()  # Synchronize all threads
            instance = ParameterUpdateLock()
            instances.append(instance)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same object
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(instance, first_instance)

    def test_initialization(self):
        """Test proper initialization of ParameterUpdateLock"""
        lock = ParameterUpdateLock()

        # Check initial state
        self.assertIsInstance(lock.param_update_lock, type(threading.RLock()))
        self.assertIsInstance(lock._attr_lock, type(threading.Lock()))
        self.assertFalse(lock._acquired)
        self.assertIsNone(lock.start_time)
        self.assertTrue(lock.param_update_completed)
        self.assertTrue(lock.first_step)

    def test_param_update_completed_property(self):
        """Test param_update_completed property getter and setter"""
        lock = ParameterUpdateLock()

        # Test initial value
        self.assertTrue(lock.param_update_completed)

        # Test setter
        lock.param_update_completed = False
        self.assertFalse(lock.param_update_completed)

        lock.param_update_completed = True
        self.assertTrue(lock.param_update_completed)

    def test_first_step_property(self):
        """Test first_step property getter and setter"""
        lock = ParameterUpdateLock()

        # Test initial value
        self.assertTrue(lock.first_step)

        # Test setter
        lock.first_step = False
        self.assertFalse(lock.first_step)

        lock.first_step = True
        self.assertTrue(lock.first_step)

    def test_acquired_property(self):
        """Test acquired property getter"""
        lock = ParameterUpdateLock()

        # Test initial value
        self.assertFalse(lock.acquired)

        # Test after manual acquisition
        lock._acquired = True
        self.assertTrue(lock.acquired)

    def test_is_healthy_initial_state(self):
        """Test is_healthy method in initial state"""
        lock = ParameterUpdateLock()

        # Initially: first_step=True, param_update_completed=True
        # Should be unhealthy because it's the first step
        self.assertFalse(lock.is_healthy())

    def test_is_healthy_after_first_step(self):
        """Test is_healthy method after first step completion"""
        lock = ParameterUpdateLock()

        # Simulate completing first step
        lock.first_step = False
        lock.param_update_completed = True

        self.assertTrue(lock.is_healthy())

    def test_is_healthy_incomplete_update(self):
        """Test is_healthy method with incomplete parameter update"""
        lock = ParameterUpdateLock()

        # Simulate incomplete parameter update
        lock.first_step = False
        lock.param_update_completed = False

        self.assertFalse(lock.is_healthy())

    def test_is_healthy_thread_safety(self):
        """Test that is_healthy is thread-safe"""
        lock = ParameterUpdateLock()
        results = []

        def check_health():
            results.append(lock.is_healthy())

        # Start multiple threads checking health
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_health)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be consistent
        self.assertEqual(len(set(results)), 1)

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_enter_main_thread_only(self, mock_main_thread, mock_current_thread):
        """Test that __enter__ can only be called from main thread"""
        lock = ParameterUpdateLock()

        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        # Should work from main thread
        with patch.object(lock, "acquire"):
            result = lock.__enter__()
            self.assertIs(result, lock)
            self.assertTrue(lock._acquired)
            self.assertFalse(lock.param_update_completed)
            self.assertIsNotNone(lock.start_time)

    def test_enter_from_any_thread(self):
        """Test that __enter__ can be called from any thread"""
        lock = ParameterUpdateLock()

        # Should work without thread restrictions
        with patch.object(lock, "acquire"):
            result = lock.__enter__()
            self.assertIs(result, lock)
            self.assertTrue(lock._acquired)
            self.assertFalse(lock.param_update_completed)
            self.assertIsNotNone(lock.start_time)

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_exit_main_thread_only(self, mock_main_thread, mock_current_thread):
        """Test that __exit__ can only be called from main thread"""
        lock = ParameterUpdateLock()

        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        # Set up state as if __enter__ was called
        lock._acquired = True
        lock.start_time = time.time()

        with patch.object(lock, "release"):
            result = lock.__exit__(None, None, None)

            self.assertFalse(result)  # Should not suppress exceptions
            self.assertFalse(lock._acquired)
            self.assertTrue(lock.param_update_completed)

    def test_exit_from_any_thread(self):
        """Test that __exit__ can be called from any thread"""
        lock = ParameterUpdateLock()

        # Set up state as if __enter__ was called
        lock._acquired = True
        lock.start_time = time.time()

        with patch.object(lock, "release"):
            result = lock.__exit__(None, None, None)

            self.assertFalse(result)  # Should not suppress exceptions
            self.assertFalse(lock._acquired)
            self.assertTrue(lock.param_update_completed)

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_exit_with_exception(self, mock_main_thread, mock_current_thread):
        """Test __exit__ behavior when exception occurred"""
        lock = ParameterUpdateLock()

        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        # Set up state as if __enter__ was called
        lock._acquired = True
        lock.start_time = time.time()

        with patch.object(lock, "release"):
            # Simulate exception
            exc_type = ValueError
            exc_val = ValueError("test error")
            exc_tb = None

            result = lock.__exit__(exc_type, exc_val, exc_tb)

            self.assertFalse(result)  # Should not suppress exceptions
            self.assertFalse(lock._acquired)
            self.assertFalse(
                lock.param_update_completed
            )  # Should be False due to exception

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_context_manager_successful_completion(
        self, mock_main_thread, mock_current_thread
    ):
        """Test context manager with successful completion"""
        lock = ParameterUpdateLock()

        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        with patch.object(lock, "acquire"), patch.object(lock, "release"):
            with lock:
                self.assertTrue(lock._acquired)
                self.assertFalse(lock.param_update_completed)

            # After context exit
            self.assertFalse(lock._acquired)
            self.assertTrue(lock.param_update_completed)

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_context_manager_with_exception(
        self, mock_main_thread, mock_current_thread
    ):
        """Test context manager when exception occurs inside"""
        lock = ParameterUpdateLock()

        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        with patch.object(lock, "acquire"), patch.object(lock, "release"):
            try:
                with lock:
                    self.assertTrue(lock._acquired)
                    self.assertFalse(lock.param_update_completed)
                    raise ValueError("test error")
            except ValueError:
                pass

            # After context exit with exception
            self.assertFalse(lock._acquired)
            self.assertFalse(lock.param_update_completed)

    def test_force_release(self):
        """Test force_release method"""
        lock = ParameterUpdateLock()

        # Mock the underlying lock
        mock_lock = MagicMock()
        lock.param_update_lock = mock_lock

        # Test successful release
        mock_lock.release.side_effect = [
            None
        ]  # First call succeeds, second raises RuntimeError
        mock_lock.release.side_effect = RuntimeError("cannot release un-acquired lock")

        lock.force_release()

        # Should have called release at least once
        mock_lock.release.assert_called()
        self.assertFalse(lock._acquired)

    def test_force_release_multiple_acquisitions(self):
        """Test force_release with multiple lock acquisitions"""
        lock = ParameterUpdateLock()

        # Mock the underlying lock to simulate multiple acquisitions
        mock_lock = MagicMock()
        lock.param_update_lock = mock_lock

        # Simulate multiple releases needed
        mock_lock.release.side_effect = [
            None,
            None,
            RuntimeError("cannot release un-acquired lock"),
        ]

        lock.force_release()

        # Should have called release until RuntimeError
        self.assertEqual(mock_lock.release.call_count, 3)
        self.assertFalse(lock._acquired)

    def test_acquire_method(self):
        """Test acquire method delegates to underlying lock"""
        lock = ParameterUpdateLock()

        # Mock the underlying lock
        mock_lock = MagicMock()
        lock.param_update_lock = mock_lock
        mock_lock.acquire.return_value = True

        result = lock.acquire()

        mock_lock.acquire.assert_called_once()
        self.assertTrue(result)

    def test_acquire_method_with_args(self):
        """Test acquire method with arguments"""
        lock = ParameterUpdateLock()

        # Mock the underlying lock
        mock_lock = MagicMock()
        lock.param_update_lock = mock_lock
        mock_lock.acquire.return_value = True

        result = lock.acquire(blocking=False, timeout=1.0)

        mock_lock.acquire.assert_called_once_with(blocking=False, timeout=1.0)
        self.assertTrue(result)

    def test_release_method(self):
        """Test release method delegates to underlying lock"""
        lock = ParameterUpdateLock()

        # Mock the underlying lock
        mock_lock = MagicMock()
        lock.param_update_lock = mock_lock

        lock.release()

        mock_lock.release.assert_called_once()

    def test_property_thread_safety(self):
        """Test that property access is thread-safe"""
        lock = ParameterUpdateLock()
        results = {"param_update_completed": [], "first_step": []}

        def access_properties():
            results["param_update_completed"].append(lock.param_update_completed)
            results["first_step"].append(lock.first_step)

        def modify_properties():
            lock.param_update_completed = False
            lock.first_step = False

        # Start multiple threads accessing and modifying properties
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=access_properties)
            threads.append(thread)
            thread = threading.Thread(target=modify_properties)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have collected results without errors
        self.assertEqual(len(results["param_update_completed"]), 5)
        self.assertEqual(len(results["first_step"]), 5)


class TestParameterUpdateLockIntegration(unittest.TestCase):
    """Integration tests for ParameterUpdateLock"""

    def setUp(self):
        """Reset singleton instance before each test"""
        ParameterUpdateLock._instance = None

    def tearDown(self):
        """Clean up after each test"""
        ParameterUpdateLock._instance = None

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_typical_training_loop_usage(self, mock_main_thread, mock_current_thread):
        """Test typical usage pattern in training loop"""
        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        lock = ParameterUpdateLock()

        # Initial state - unhealthy (first step)
        self.assertFalse(lock.is_healthy())

        # First training step
        with lock:
            # Inside critical section
            self.assertFalse(lock.param_update_completed)
            # Simulate parameter update
            pass

        # After first step
        self.assertTrue(lock.param_update_completed)
        lock.first_step = False  # Simulate completion of first step

        # Now should be healthy
        self.assertTrue(lock.is_healthy())

        # Subsequent training steps
        for step in range(3):
            with lock:
                self.assertFalse(lock.param_update_completed)
                # Simulate parameter update
                pass

            # After each step
            self.assertTrue(lock.param_update_completed)
            self.assertTrue(lock.is_healthy())

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_training_loop_with_exception(self, mock_main_thread, mock_current_thread):
        """Test training loop when exception occurs during parameter update"""
        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        lock = ParameterUpdateLock()
        lock.first_step = False  # Simulate past first step

        # Initially healthy
        self.assertTrue(lock.is_healthy())

        # Training step with exception
        try:
            with lock:
                self.assertFalse(lock.param_update_completed)
                raise RuntimeError("Parameter update failed")
        except RuntimeError:
            pass

        # After exception, should be unhealthy
        self.assertFalse(lock.param_update_completed)
        self.assertFalse(lock.is_healthy())

    def test_concurrent_health_checks(self):
        """Test concurrent health checks from multiple threads"""
        lock = ParameterUpdateLock()
        lock.first_step = False
        lock.param_update_completed = True

        results = []
        barrier = threading.Barrier(10)

        def check_health():
            barrier.wait()  # Synchronize all threads
            results.append(lock.is_healthy())

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_health)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All health checks should return the same result
        self.assertTrue(all(results))
        self.assertEqual(len(results), 10)

    def test_force_release_during_acquisition(self):
        """Test force_release called while lock is acquired"""
        lock = ParameterUpdateLock()

        # Acquire the lock normally
        lock.acquire()

        # Force release should clear the lock
        lock.force_release()

        # Lock should be available for acquisition
        acquired = lock.acquire(blocking=False)
        self.assertTrue(acquired)
        lock.release()

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_timing_measurement(self, mock_main_thread, mock_current_thread):
        """Test that timing is properly measured"""
        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        lock = ParameterUpdateLock()

        start_time = time.time()
        with lock:
            time.sleep(0.01)  # Small delay
        end_time = time.time()

        # Should have measured some time
        self.assertIsNotNone(lock.start_time)
        # The measured duration should be reasonable
        duration = end_time - start_time
        self.assertGreater(duration, 0.005)  # At least 5ms


class TestParameterUpdateLockCrossThread(unittest.TestCase):
    """Tests for cross-thread parameter lock usage after removing thread safety assertions"""

    def setUp(self):
        """Reset singleton instance before each test"""
        ParameterUpdateLock._instance = None

    def tearDown(self):
        """Clean up after each test"""
        ParameterUpdateLock._instance = None

    def test_context_manager_from_worker_thread(self):
        """Test that parameter lock can be used from worker threads"""
        lock = ParameterUpdateLock()
        result = []
        
        def worker_thread_task():
            """Task that runs in worker thread and uses parameter lock"""
            try:
                with lock:
                    # Should work without thread restrictions now
                    result.append("worker_success")
                    result.append(lock._acquired)  # Should be True while in context
                result.append(lock._acquired)  # Should be False after context exits
            except Exception as e:
                result.append(f"error: {e}")
        
        # Run in worker thread
        worker_thread = threading.Thread(target=worker_thread_task)
        worker_thread.start()
        worker_thread.join()
        
        # Verify it worked from worker thread
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "worker_success")
        self.assertTrue(result[1])  # _acquired was True inside context
        self.assertFalse(result[2])  # _acquired is False after context

    def test_fault_handling_thread_usage(self):
        """Test parameter lock usage from fault handling thread context"""
        lock = ParameterUpdateLock()
        fault_handler_result = []
        
        def simulate_fault_handler():
            """Simulate fault handling thread lock actions"""
            # Check lock state (this should work from any thread)
            is_healthy = lock.is_healthy()
            fault_handler_result.append(f"healthy: {is_healthy}")
            
            # Access acquired property (this should work from any thread) 
            is_acquired = lock.acquired
            fault_handler_result.append(f"acquired: {is_acquired}")
        
        # Simulate fault handling from separate thread
        fault_thread = threading.Thread(target=simulate_fault_handler)
        fault_thread.start()
        fault_thread.join()
        
        # Verify fault handler could interact with lock
        self.assertEqual(len(fault_handler_result), 2)
        self.assertIn("healthy: False", fault_handler_result)  # Initially unhealthy (first_step=True)
        self.assertIn("acquired: False", fault_handler_result)  # Initially not acquired


class TestParameterUpdateLockEdgeCases(unittest.TestCase):
    """Edge case tests for ParameterUpdateLock"""

    def setUp(self):
        """Reset singleton instance before each test"""
        ParameterUpdateLock._instance = None

    def tearDown(self):
        """Clean up after each test"""
        ParameterUpdateLock._instance = None

    def test_multiple_singleton_creations_concurrent(self):
        """Test multiple concurrent singleton creations"""
        instances = []
        errors = []

        def create_and_use():
            try:
                lock = ParameterUpdateLock()
                instances.append(lock)
                # Try to use the lock
                lock.param_update_completed = False
                lock.param_update_completed = True
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(20):
            thread = threading.Thread(target=create_and_use)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(instance, first_instance)

    def test_force_release_when_not_acquired(self):
        """Test force_release when lock is not acquired"""
        lock = ParameterUpdateLock()

        # Should not raise exception
        lock.force_release()
        self.assertFalse(lock._acquired)

    def test_property_access_during_modification(self):
        """Test property access while another thread is modifying"""
        lock = ParameterUpdateLock()

        def continuous_modification():
            for _ in range(100):
                lock.param_update_completed = not lock.param_update_completed
                lock.first_step = not lock.first_step
                time.sleep(0.001)

        def continuous_access():
            for _ in range(100):
                _ = lock.param_update_completed
                _ = lock.first_step
                _ = lock.is_healthy()
                time.sleep(0.001)

        modifier_thread = threading.Thread(target=continuous_modification)
        accessor_thread = threading.Thread(target=continuous_access)

        modifier_thread.start()
        accessor_thread.start()

        modifier_thread.join()
        accessor_thread.join()

        # Should complete without deadlock or exception

    @patch("threading.current_thread")
    @patch("threading.main_thread")
    def test_nested_context_manager_usage(self, mock_main_thread, mock_current_thread):
        """Test nested usage of context manager (should work with RLock)"""
        # Mock main thread
        main_thread = MagicMock()
        mock_main_thread.return_value = main_thread
        mock_current_thread.return_value = main_thread

        lock = ParameterUpdateLock()

        def inner_function():
            with lock:
                self.assertTrue(lock._acquired)
                return "inner"

        with lock:
            self.assertTrue(lock._acquired)
            result = inner_function()
            self.assertEqual(result, "inner")
            # After inner context exits, _acquired becomes False even though outer context is still active
            # This is because _acquired is a simple boolean, not a counter
            self.assertFalse(lock._acquired)

        self.assertFalse(lock._acquired)
        self.assertTrue(lock.param_update_completed)


if __name__ == "__main__":
    unittest.main()
