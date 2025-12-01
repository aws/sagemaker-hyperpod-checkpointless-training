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

import pytest
import copy
from unittest.mock import Mock, patch

from hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer import (
    StartupOverheadTracer,
    NUM_STEPS_TO_TRACE,
    TRACING_PROFILES_PATH,
    STARTUP_VIZTRACER_ENTRIES,
    VIZTRACER_MIN_DURATION,
)


class TestStartupOverheadTracer:
    """Test StartupOverheadTracer class"""

    @patch('hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer.VizTracer')
    def test_initialization(self, mock_viztracer_class):
        """Test StartupOverheadTracer initialization"""
        mock_tracer = Mock()
        mock_viztracer_class.return_value = mock_tracer
        
        tracer = StartupOverheadTracer()
        
        # Verify VizTracer was created with correct parameters
        mock_viztracer_class.assert_called_once_with(
            min_duration=VIZTRACER_MIN_DURATION,
            tracer_entries=STARTUP_VIZTRACER_ENTRIES,
            exclude_files=[
                "*/triton/*",
                "*/transformer_engine/*",
                "*/torch/_inductor/*",
            ],
            ignore_c_function=True
        )
        
        # Verify tracer was started and initial state is correct
        mock_tracer.start.assert_called_once()
        assert tracer.current_step == 0
        assert tracer.is_stopped is False

    @patch('hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer.VizTracer')
    def test_on_train_batch_start_increments_step(self, mock_viztracer_class):
        """Test on_train_batch_start increments step counter"""
        mock_tracer = Mock()
        mock_viztracer_class.return_value = mock_tracer
        
        tracer = StartupOverheadTracer()
        mock_trainer = Mock()
        mock_pl_module = Mock()
        mock_batch = Mock()
        
        tracer.on_train_batch_start(mock_trainer, mock_pl_module, mock_batch, 0)
        
        assert tracer.current_step == 1
        assert tracer.is_stopped is False
        mock_tracer.stop.assert_not_called()

    @patch('hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer.VizTracer')
    def test_tracer_basic_functionality(self, mock_viztracer_class):
        """Test basic tracer functionality without complex stopping logic"""
        mock_tracer = Mock()
        mock_viztracer_class.return_value = mock_tracer
        
        tracer = StartupOverheadTracer()
        
        # Test that tracer starts correctly
        mock_tracer.start.assert_called_once()
        assert tracer.current_step == 0
        assert tracer.is_stopped is False
        
        # Test step increment
        tracer.on_train_batch_start(Mock(), Mock(), Mock(), 0)
        assert tracer.current_step == 1
        
        # Test that tracer can be stopped manually
        tracer.is_stopped = True
        tracer.on_train_batch_start(Mock(), Mock(), Mock(), 0)
        # Should not increment when stopped
        assert tracer.current_step == 1

    @patch('hyperpod_checkpointless_training.inprocess.tools.startup_overhead_tracer.VizTracer')
    def test_deepcopy_excludes_tracer(self, mock_viztracer_class):
        """Test __deepcopy__ method excludes tracer to avoid pickling issues"""
        mock_tracer = Mock()
        mock_viztracer_class.return_value = mock_tracer
        
        tracer = StartupOverheadTracer()
        tracer.current_step = 5
        
        copied_tracer = copy.deepcopy(tracer)
        
        # Verify basic attributes were copied but tracer was excluded
        assert copied_tracer.current_step == 5
        assert copied_tracer.tracer is None
        assert tracer.tracer == mock_tracer  # Original unchanged
