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

import os
import threading
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

from hyperpod_checkpointless_training.inprocess.hp_fault_handling_thread import HPFaultHandlingThread


class TestHPFaultHandlingThread(unittest.TestCase):
    def setUp(self):
        os.environ["RANK"] = "0"
        os.environ["HPWRAPPER_LOG_LEVEL"] = "debug"  # test with full logging
        self.failure_event = threading.Event()
        self.stop_raising_event = threading.Event()
        self.stop_raising_event.set()
        self.atomic_lock = threading.RLock()
        
        # Create mock abort instances
        self.mock_dataloader_abort = Mock()
        self.mock_communicator_abort = Mock()
        self.mock_checkpointing_abort = Mock()
        
        # Add post_comm_abort_cleanup method to checkpointing abort
        self.mock_checkpointing_abort.post_comm_abort_cleanup = Mock()
        
        # Create a mock abort object with instances attribute
        self.mock_abort = Mock()
        self.mock_abort.instances = [
            self.mock_dataloader_abort,
            self.mock_communicator_abort,
            self.mock_checkpointing_abort
        ]

        self.thread = HPFaultHandlingThread(
            failure=self.failure_event,
            stop_raising=self.stop_raising_event,
            atomic_lock=self.atomic_lock,
            abort=self.mock_abort,
            abort_timeout=10.0,
            soft_timeout=30.0,
            hard_timeout=30.0,
            failure_check_interval=1.0,
            async_raise_before_abort=True,
            early_abort_communicator=False,
            seq=Mock()
        )
        self.mock_async_abort = Mock()
        self.thread.async_abort_main_thread = self.mock_async_abort

    def test_initialization(self):
        """Test proper initialization of HPFaultHandlingThread"""
        self.assertEqual(self.thread.state.rank, 0)
        self.assertEqual(self.thread.failure, self.failure_event)
        self.assertEqual(self.thread.stop_raising, self.stop_raising_event)
        self.assertEqual(self.thread.atomic_lock, self.atomic_lock)
        self.assertIsInstance(self.thread.should_stop, threading.Event)

    def test_handle_failure(self):
        """Test handle_failure method"""
        self.thread.handle_failure()
        self.mock_abort.assert_called_once()

    def test_do_spin_main_abort(self):
        """Test do_spin_main_abort method"""
        self.stop_raising_event.clear()

        def stop_thread():
            time.sleep(0.1)
            self.thread.should_stop.set()

        # Start a thread to stop the spin
        stop_thread_thread = threading.Thread(target=stop_thread)
        stop_thread_thread.start()

        self.thread.do_spin_main_abort()
        self.assertTrue(self.mock_async_abort.called)
        self.stop_raising_event.set()

    def test_shutdown(self):
        """Test shutdown method"""
        self.thread.shutdown()
        self.assertTrue(self.thread.should_stop.is_set())

    def test_run_with_failure(self):
        """Test run method when failure occurs"""

        def trigger_failure():
            time.sleep(0.1)
            self.failure_event.set()

        # Start a thread to trigger failure
        trigger_thread = threading.Thread(target=trigger_failure)
        trigger_thread.start()

        self.thread.run()
        self.mock_async_abort.assert_called()

    def test_run_with_shutdown(self):
        """Test run method when shutdown is called"""

        def do_shutdown():
            time.sleep(0.1)
            self.thread.shutdown()

        # Start a thread to shutdown
        shutdown_thread = threading.Thread(target=do_shutdown)
        shutdown_thread.start()

        self.thread.run()
        self.assertTrue(self.thread.should_stop.is_set())

    def test_do_main_abort(self):
        """Test do_main_abort method"""
        self.thread.do_main_abort()
        self.mock_async_abort.assert_called_once()

    def test_do_abort_with_exception(self):
        """Test do_abort method when abort raises exception"""
        self.mock_abort.side_effect = Exception("Abort error")
        self.thread.do_abort()  # Should not raise exception
        self.mock_abort.assert_called_once()

    def test_try_abort_with_early_communicator_abort(self):
        """Test try_abort when early_abort_communicator=True"""
        # Set up thread with early_abort_communicator=True
        self.thread.early_abort_communicator = True
        self.thread.async_raise_before_abort = True

        # Create a list to track the order of calls
        call_order = []
        self.thread.do_main_abort = Mock(
            side_effect=lambda: call_order.append("main_abort")
        )
        self.thread.do_abort = Mock(
            side_effect=lambda: call_order.append("abort")
        )
        self.thread.do_post_comm_abort_cleanup = Mock(
            side_effect=lambda: call_order.append("post_comm_abort_cleanup")
        )

        # Mock reorder_aborts to verify it's called
        self.thread.reorder_aborts = Mock(return_value=self.mock_abort)

        self.thread.try_abort()

        # Verify the order of calls
        self.assertEqual(
            call_order, ["main_abort", "abort", "post_comm_abort_cleanup"]
        )
        self.thread.do_main_abort.assert_called_once()
        self.thread.do_abort.assert_called_once()
        self.thread.do_post_comm_abort_cleanup.assert_called_once()
        # Verify reorder_aborts was called with the correct first parameter type
        self.thread.reorder_aborts.assert_called_once()
        args, _ = self.thread.reorder_aborts.call_args
        self.assertEqual(args[1], 0)  # Check that target_index is 0

    def test_try_abort_without_early_communicator_abort(self):
        """Test try_abort when early_abort_communicator=False"""
        # Set up thread with early_abort_communicator=False (default)
        self.thread.early_abort_communicator = False
        self.thread.async_raise_before_abort = True

        # Create a list to track the order of calls
        call_order = []
        self.thread.do_main_abort = Mock(
            side_effect=lambda: call_order.append("main_abort")
        )
        self.thread.do_abort = Mock(
            side_effect=lambda: call_order.append("abort")
        )
        self.thread.do_post_comm_abort_cleanup = Mock(
            side_effect=lambda: call_order.append("post_comm_abort_cleanup")
        )

        self.thread.try_abort()

        # Verify the order of calls
        self.assertEqual(
            call_order, ["main_abort", "abort", "post_comm_abort_cleanup"]
        )
        self.thread.do_main_abort.assert_called_once()
        self.thread.do_abort.assert_called_once()
        self.thread.do_post_comm_abort_cleanup.assert_called_once()

    def test_do_post_comm_abort_cleanup_with_instances(self):
        """Test do_post_comm_abort_cleanup when abort has instances with post_comm_abort_cleanup"""
        self.thread.do_post_comm_abort_cleanup()
        
        # Verify post_comm_abort_cleanup was called on the instance that has it
        self.mock_checkpointing_abort.post_comm_abort_cleanup.assert_called_once()

    def test_do_post_comm_abort_cleanup_without_abort(self):
        """Test do_post_comm_abort_cleanup when abort is None"""
        self.thread.abort = None
        
        # Should not raise exception when abort is None
        self.thread.do_post_comm_abort_cleanup()

    def test_reorder_aborts(self):
        """Test reorder_aborts method"""
        # Create mock instances
        mock_dataloader = Mock()
        mock_comm = Mock()
        mock_checkpoint = Mock()
        
        # Create a mock abort with instances
        mock_abort = Mock()
        mock_abort.instances = [mock_comm, mock_dataloader, mock_checkpoint]
        
        # Save the original abort and replace with our mock
        original_abort = self.thread.abort
        self.thread.abort = mock_abort
        
        # Mock the Compose class/function that would be returned
        with patch('hyperpod_checkpointless_training.inprocess.hp_fault_handling_thread.Compose', return_value=mock_abort) as mock_compose:
            # Test reordering to move an instance to index 0
            # We'll use the mock_dataloader's class as the instance_type
            result = self.thread.reorder_aborts(mock_dataloader.__class__, 0)
            
            # Verify the result is our mock_abort
            self.assertEqual(result, mock_abort)
        
        # Restore the original abort
        self.thread.abort = original_abort
        
        # Test with invalid target index
        with self.assertRaises(ValueError):
            self.thread.reorder_aborts(mock_dataloader.__class__, 10)


if __name__ == "__main__":
    unittest.main()
