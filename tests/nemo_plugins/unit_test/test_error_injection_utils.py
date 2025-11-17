import unittest
from unittest.mock import MagicMock, patch
import torch
import time
import os
import signal
from hyperpod_checkpointless_training.inprocess.tools.inject_fault import Fault

from hyperpod_checkpointless_training.nemo_plugins.error_injection_utils import (
    TestFaultConfig,
    ErrorInjectionHook,
    ErrorInjectionForwardHook,
    ErrorInjectionBackwardHook,
    RandomErrorInjector,
    find_middle_transformer_layer,
    inject_fault,
    UNSUPPORTED_RANDOM_FAULT_INJECTION,
    FaultInjector,
)

class TestErrorInjectionModule(unittest.TestCase):
    def setUp(self):
        # Setup common mocks
        self.trainer_mock = MagicMock()
        self.module_mock = MagicMock()

        # Mock trainer attributes
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop.hp_wrapper.seq.get.return_value = 1
        self.trainer_mock.global_step = 20
        self.trainer_mock.fit_loop.hp_wrapper.step_upon_restart = 20
        self.trainer_mock.global_step = 42

        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'WORLD_SIZE': '4',
            'RANK': '0',
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_test_fault_config_init(self):
        """Test initialization of TestFaultConfig with default and custom values."""
        # Default initialization
        config = TestFaultConfig()
        self.assertEqual(config.fault_type, "ipr")
        self.assertEqual(config.fault_prob_after_bwd, 0)
        self.assertEqual(config.steps_before_fault, 10000)
        self.assertEqual(config.fault_prob_after_bwd, 0)
        self.assertEqual(config.fault_prob_between_lock, 0)
        self.assertEqual(config.fault_prob_during_fwd, 0)
        self.assertEqual(config.fault_prob_during_bwd, 0)
        self.assertEqual(config.fault_ranks, set())
        self.assertEqual(config.steps_before_fault, 10000)
        self.assertEqual(config.fault_time_interval_sec, -1)

        # Custom initialization
        custom_config = TestFaultConfig(
            fault_type="plr",
            fault_prob_after_bwd=0.5,
            fault_ranks={1, 2},
            steps_before_fault=100
        )
        self.assertEqual(custom_config.fault_type, "plr")
        self.assertEqual(custom_config.fault_prob_after_bwd, 0.5)
        self.assertEqual(custom_config.fault_ranks, {1, 2})
        self.assertEqual(custom_config.steps_before_fault, 100)

    def test_test_fault_config_validation(self):
        """Test validation in TestFaultConfig post-init."""
        with patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.hp_logger") as mock_logger:
            test_cases = [
                ("invalid_fault_type", dict(fault_type="invalid"),
                f"Invalid fault_type: invalid. Must be 'ipr' or 'plr' or {[fault.name for fault in Fault]}. "
                f"Note: random fault injector does not support the following fault types: {UNSUPPORTED_RANDOM_FAULT_INJECTION}.",
                dict(fault_type="invalid"),
                ),

                ("prob_after_bwd_out_of_bounds", dict(fault_prob_after_bwd=1.5),
                "prob_after_bwd must be between 0 and 1, got 1.5",
                dict(fault_prob_after_bwd=0),
                ),

                ("prob_between_lock_out_of_bounds", dict(fault_prob_between_lock=-0.1),
                "prob_between_lock must be between 0 and 1, got -0.1",
                dict(fault_prob_between_lock=0),
                ),

                ("prob_during_fwd_out_of_bounds", dict(fault_prob_during_fwd=1.1),
                "fault_prob_during_fwd must be between 0 and 1, got 1.1",
                dict(fault_prob_during_fwd=0),
                ),

                ("prob_during_bwd_out_of_bounds", dict(fault_prob_during_bwd=-0.1),
                "fault_prob_during_bwd must be between 0 and 1, got -0.1",
                dict(fault_prob_during_bwd=0),
                ),

                ("prob_random_out_of_bounds", dict(fault_prob_random=2.0),
                "fault_prob_random must be between 0 and 1, got 2.0",
                dict(fault_prob_random=0),
                ),

                ("fault_time_interval_sec_positive_resets_steps_before_fault",
                dict(fault_time_interval_sec=30, steps_before_fault=100),
                "Time-based fault injection mode enabled. Resetting steps_before_fault to -1 to disable step-based fault injection mode.",
                dict(fault_time_interval_sec=30, steps_before_fault=-1)
                ),

                ("fault_time_interval_sec_negative_keeps_steps_before_fault",
                dict(fault_time_interval_sec=-5, steps_before_fault=100),
                None,
                dict(fault_time_interval_sec=-5, steps_before_fault=100)
                ),
            ]

            for name, kwargs, expected_warning, expected_values in test_cases:
                with self.subTest(name=name):
                    mock_logger.reset_mock()
                    config = TestFaultConfig(**kwargs)
                    for key, expected_val in expected_values.items():
                        actual_val = getattr(config, key)
                        self.assertEqual(
                            actual_val, expected_val,
                            f"Expected {key}={expected_val}, but got {actual_val} in case '{name}'"
                        )
                    if expected_warning:
                        matched = any(expected_warning in str(call.args[0]) for call in mock_logger.warning.call_args_list)
                        self.assertTrue(matched, f"Expected warning containing '{expected_warning}' not found for case '{name}'")
                    else:
                        mock_logger.warning.assert_not_called()

    def test_error_injection_hook_init(self):
        """Test initialization of ErrorInjectionHook."""
        config = TestFaultConfig()
        fault_injector_mock = MagicMock()
        hook = ErrorInjectionHook(config, self.trainer_mock, fault_injector_mock)
        self.assertEqual(hook.test_fault_config, config)
        self.assertEqual(hook.trainer, self.trainer_mock)
        self.assertEqual(hook.rank, 0)
        self.assertEqual(hook.current_step, -1)
        self.assertEqual(hook.fault_injector, fault_injector_mock)

    def test_error_injection_forward_hook(self):
        """Test ErrorInjectionForwardHook."""
        config = TestFaultConfig(fault_prob_during_fwd=1.0)
        fault_injector = FaultInjector(config, self.trainer_mock)
        hook = ErrorInjectionForwardHook(config, self.trainer_mock, fault_injector)

        with patch.object(fault_injector, 'maybe_inject') as mock_maybe_inject:
            output = torch.tensor([1.0])
            hook(self.module_mock, None, output)
            mock_maybe_inject.assert_called_once_with(
                fault_prob=config.fault_prob_during_fwd,
                msg="Simulating failure during forward pass",
            )

    def test_error_injection_backward_hook(self):
        """Test ErrorInjectionBackwardHook."""
        config = TestFaultConfig(fault_prob_during_bwd=1.0)
        fault_injector = FaultInjector(config, self.trainer_mock)
        hook = ErrorInjectionBackwardHook(config, self.trainer_mock, fault_injector)

        with patch.object(fault_injector, 'maybe_inject') as mock_maybe_inject:
            grad_input = (torch.tensor([1.0]),)
            hook(self.module_mock, grad_input, None)
            mock_maybe_inject.assert_called_once_with(
                fault_prob=config.fault_prob_during_bwd,
                msg="Simulating failure during backward pass",
            )

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.inject_fault")
    @patch("random.random")
    def test_fault_injector_maybe_inject_step_based_injection_mode(self, mock_random, mock_inject_fault):
        """Test maybe_inject method."""
        test_cases = [
            # fault_type, random_val, step_upon_restart, rank, should_raise, expected_exception
            ("ipr", 0.4, 20, 0, True, RuntimeError),
            ("ipr", 0.6, 20, 0, False, None),
            ("ipr", 0.4, 5, 0, False, None),
            ("ipr", 0.4, 20, 2, False, None),
            ("plr", 0.4, 20, 0, True, SystemExit),
            ("plr", 0.6, 20, 0, False, None),
            ("GPU_ERROR", 0.4, 20, 0, False, None),
            ("GPU_ERROR", 0.6, 20, 0, False, None),
        ]

        for fault_type, random_val, step_upon_restart, rank, should_raise, expected_exception in test_cases:
            with self.subTest(fault_type=fault_type, random_val=random_val, 
                              step_upon_restart=step_upon_restart, rank=rank, should_raise=should_raise):
                mock_random.return_value = random_val
                fault_prob_after_bwd = 0.5
                config = TestFaultConfig(
                    fault_type=fault_type,
                    fault_prob_after_bwd=fault_prob_after_bwd,
                    fault_ranks={0, 1},
                    steps_before_fault=10
                )
                self.trainer_mock.strategy.cluster_environment.global_rank.return_value = rank
                self.trainer_mock.strategy.get_wrapper.return_value.step_upon_restart = step_upon_restart = step_upon_restart

                injector = FaultInjector(config, self.trainer_mock)
                injector.last_injection_time = time.time() - 100  # ensure interval elapsed

                if fault_type == "plr":
                    with patch('os.kill', side_effect=SystemExit):
                        if should_raise:
                            with self.assertRaises(expected_exception):
                                injector.maybe_inject(fault_prob_after_bwd, msg="Test error")
                        else:
                            injector.maybe_inject(fault_prob_after_bwd, msg="Test error")
                elif should_raise:
                    with self.assertRaises(expected_exception):
                        injector.maybe_inject(fault_prob_after_bwd, msg="Test error")
                else:
                    injector.maybe_inject(fault_prob_after_bwd, msg="Test error")

                # Verify inject_fault is called only for non-ipr/plr
                if fault_type not in ("ipr", "plr") and random_val < fault_prob_after_bwd and rank in config.fault_ranks and step_upon_restart > config.steps_before_fault:
                    mock_inject_fault.assert_called_once_with(fault_type)
                else:
                    mock_inject_fault.assert_not_called()

                mock_inject_fault.reset_mock()  # clean slate for next case

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.dispatch_fault_injection")
    def test_inject_fault_with_valid_type(self, mock_dispatch):
        for fault in Fault:
            with self.subTest(fault_type=fault.name):
                mock_dispatch.reset_mock()
                inject_fault(fault.name)
                mock_dispatch.assert_called_once_with(
                    fault=fault,
                    delay=0.0,
                    callback=unittest.mock.ANY,
                )

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.hp_logger")
    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.dispatch_fault_injection")
    def test_inject_fault_with_invalid_type(self, mock_dispatch, mock_logger):
        invalid_fault_type = "INVALID_FAULT_TYPE"
        inject_fault(invalid_fault_type)
        mock_dispatch.assert_not_called()
        mock_logger.warning.assert_called_once_with(
            f"Undefined fault_type: {invalid_fault_type}"
        )

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.dispatch_fault_injection")
    def test_inject_fault_with_callback(self, mock_dispatch):
        callback = MagicMock()
        inject_fault("GPU_ERROR", callable=callback)
        callback.assert_called_once()
        mock_dispatch.assert_called_once()

    def test_find_middle_transformer_layer(self):
        """Test find_middle_transformer_layer function."""
        class TransformerLayer(torch.nn.Module):
            pass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([TransformerLayer() for _ in range(3)])

        model = Model()
        middle_layer = find_middle_transformer_layer(model)
        self.assertIsInstance(middle_layer, TransformerLayer)
        self.assertEqual(middle_layer, model.layers[1])

        # Test with no TransformerLayer
        linear_model = torch.nn.Linear(10, 10)
        self.assertIsNone(find_middle_transformer_layer(linear_model))


class TestFaultInjector(unittest.TestCase):
    def setUp(self):
        self.mock_trainer = MagicMock()
        mock_wrapper = MagicMock()
        mock_seq = MagicMock()
        mock_seq.get.return_value = 5
        mock_wrapper.seq = mock_seq
        mock_wrapper.step_upon_restart = 42
        
        self.mock_trainer.strategy.get_wrapper.return_value = mock_wrapper
        self.mock_trainer.strategy.cluster_environment.global_rank.return_value = 0
        self.mock_trainer.global_step = 42

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.random.random", return_value=0.1)
    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.time.time")
    def test_maybe_inject_time_and_step_modes(self, mock_time, mock_random):
        now = 1000
        mock_time.return_value = now
        fault_prob = 0.5  # random.random() returns 0.1 < 0.5

        test_cases = [
            # Time-based mode (fault_time_interval_sec >= 0), step_mode_trigger always False
            # (fault_time_interval_sec, last_injection_time offset, step_upon_restart, expected_raise, description)
            (30, now - 40, 2, True, "time_mode_trigger=True, step_mode_trigger=False"),   # time_lapsed=40 > 30
            (30, now - 10, 10, False, "time_mode_trigger=False, step_mode_trigger=False"), # time_lapsed=10 < 30

            # Step-based mode (fault_time_interval_sec < 0), time_mode_trigger always False
            ( -1, now - 10, 10, True, "time_mode_trigger=False, step_mode_trigger=True"),  # step_upon_restart=10 > 5 threshold
            ( -1, now - 10, 2, False, "time_mode_trigger=False, step_mode_trigger=False"), # step_upon_restart=2 < 5 threshold
        ]

        for fault_time_interval_sec, last_injection_time, step_upon_restart, should_raise, desc in test_cases:
            with self.subTest(desc=desc):
                config = TestFaultConfig(
                    fault_type="ipr",
                    fault_ranks={0},
                    fault_time_interval_sec=fault_time_interval_sec,
                    steps_before_fault=5,
                )
                injector = FaultInjector(config, self.mock_trainer)
                injector.last_injection_time = last_injection_time
                self.mock_trainer.strategy.get_wrapper.return_value.step_upon_restart = step_upon_restart

                if should_raise:
                    with self.assertRaises(RuntimeError):
                        injector.maybe_inject(fault_prob, msg=f"Test ipr with {desc}")
                else:
                    # No exception expected
                    injector.maybe_inject(fault_prob, msg=f"Test ipr with {desc}")

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.debug_msg")
    @patch("random.random", return_value=0.0)  # force prob < fault_prob
    @patch("time.time", return_value=100)  # fixed "now"
    def test_maybe_inject_no_hp_wrapper(self, mock_time, mock_random, mock_debug_msg):
        config = TestFaultConfig(
                    fault_type="ipr",
                    fault_ranks={0},
                    fault_time_interval_sec=0, # time-based mode
                    steps_before_fault=-1,
                )

        self.mock_trainer.strategy.get_wrapper.return_value = None
        fault_injector = FaultInjector(config, self.mock_trainer)
        fault_injector.last_injection_time = 0  # ensure time_lapsed = 100 > 10

        with self.assertRaises(RuntimeError):
            fault_injector.maybe_inject(1.0, msg="Test message")

        mock_debug_msg.assert_called_once_with(
            "[TIME_MODE]: Test message",
            rank=0,  # Use concrete value
            seq=0,   # Should be 0 when no wrapper
            steps=42  # Use concrete value
        )

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.debug_msg")
    @patch("random.random", return_value=0.0)
    @patch("time.time", return_value=100)
    def test_maybe_inject_with_hp_wrapper_seq(self, mock_time, mock_random, mock_debug_msg):
        config = TestFaultConfig(
            fault_type="ipr",
            fault_ranks={0},
            fault_time_interval_sec=0,
            steps_before_fault=-1,
        )

        fault_injector = FaultInjector(config, self.mock_trainer)
        fault_injector.last_injection_time = 0

        with self.assertRaises(RuntimeError):
            fault_injector.maybe_inject(1.0, msg="Test message")

        mock_debug_msg.assert_called_once_with(
            "[TIME_MODE]: Test message",
            rank=0,  # Using concrete value
            seq=5,   # Using concrete value that matches setUp
            steps=42 # Using concrete value
        )

@patch('time.sleep')  # Patch sleep to avoid waiting during tests
class TestRandomErrorInjector(unittest.TestCase):
    def setUp(self):
        # Setup common mocks
        self.trainer_mock = MagicMock()
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.strategy.get_wrapper.return_value.step_upon_restart = 20
        self.trainer_mock.global_step = 20

        self.test_fault_config = TestFaultConfig(
            fault_type="ipr",
            fault_prob_random=0.5,
            fault_ranks={0, 1},
            steps_before_fault=10
        )

    def test_init(self, _):
        """Test initializatiation of RandomErrorInjector."""
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        self.assertEqual(injector.test_fault_config, self.test_fault_config)
        self.assertEqual(injector.trainer, self.trainer_mock)
        self.assertEqual(injector.rank, 0)
        self.assertTrue(injector.daemon)
        self.assertFalse(injector.stop_event.is_set())

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_run_ipr_fault(self, mock_kill, mock_random, mock_uniform, _):
        """Test run method with IPR fault injection."""
        mock_uniform.return_value = 5.0  # Sleep duration
        mock_random.return_value = 0.4  # Below fault_prob_random threshold

        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_kill.assert_called_once_with(os.getpid(), signal.SIGUSR1)

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_run_plr_fault(self, mock_kill, mock_random, mock_uniform, _):
        """Test run method with PLR fault injection."""
        mock_uniform.return_value = 5.0
        mock_random.return_value = 0.4

        self.test_fault_config.fault_type = "plr"
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_kill.assert_called_once_with(os.getpid(), signal.SIGKILL)

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.inject_fault")
    @patch("random.random", return_value=0.4)
    @patch("random.uniform", return_value=5.0)
    def test_run_nvrx_sigterm_fault(self, mock_uniform, mock_random, mock_inject_fault, _):
        """Test run method with NVRx fault injection."""
        # Define side effect to stop thread after first fault injection since inject_fault
        # with the stop callback is mocked, the thread doesn't stop so it will continue
        # looping unless we force it to in this test
        def inject_and_stop(fault_type, stop_cb):
            stop_cb()

        mock_inject_fault.side_effect = inject_and_stop

        self.test_fault_config.fault_type = "SIGTERM"
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_inject_fault.assert_called_once_with("SIGTERM", injector.stop)

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_no_fault_injection_probability(self, mock_kill, mock_random, mock_uniform, _):
        """Test when random value is above fault probability threshold."""
        mock_uniform.return_value = 5.0
        mock_random.return_value = 0.6  # Above fault_prob_random threshold

        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_kill.assert_not_called()

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_no_fault_injection_steps(self, mock_kill, mock_random, mock_uniform, _):
        """Test when steps_upon_restart is below threshold."""
        mock_uniform.return_value = 5.0
        mock_random.return_value = 0.4

        self.trainer_mock.strategy.get_wrapper.return_value.step_upon_restart = 5 # Below steps_before_fault
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_kill.assert_not_called()

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_no_fault_injection_wrong_rank(self, mock_kill, mock_random, mock_uniform, _):
        """Test when rank is not in fault_ranks."""
        mock_uniform.return_value = 5.0
        mock_random.return_value = 0.4

        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 2  # Not in fault_ranks
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_kill.assert_not_called()

    @patch("random.uniform")
    @patch("random.random")
    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.inject_fault")
    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.hp_logger")
    def test_random_injector_unsupported_fault_type(self, mock_logger, mock_inject, mock_random, mock_uniform, _):
        mock_random.return_value = 0.1
        mock_uniform.return_value = 0.1
        self.test_fault_config.fault_type = "SIGNAL_EXC"  # In unsupported list

        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)
        injector.start()
        time.sleep(0.1)
        injector.stop()
        injector.join()

        mock_logger.warning.assert_called_with("Unsupported fault_type: SIGNAL_EXC")
        mock_inject.assert_not_called()

    def test_stop(self, _):
        """Test stop method."""
        injector = RandomErrorInjector(self.test_fault_config, self.trainer_mock)

        self.assertFalse(injector.stop_event.is_set())
        injector.stop()
        self.assertTrue(injector.stop_event.is_set())

    @patch('random.uniform')
    @patch('random.random')
    @patch('os.kill')
    def test_run_sleep_duration(self, mock_kill, mock_random, mock_uniform, mock_sleep):
        test_cases = [
            {
                "desc": "Fixed sleep mode (fault_time_interval_sec >= 0)",
                "fault_time_interval_sec": 10,
                "expected_sleep": 10,
                "mock_uniform_return": None,
            },
            {
                "desc": "Random sleep mode (fault_time_interval_sec < 0)",
                "fault_time_interval_sec": -1,
                "expected_sleep": 7.5,
                "mock_uniform_return": 7.5,
            },
        ]

        for case in test_cases:
            with self.subTest(case["desc"]):
                config = TestFaultConfig(
                    fault_type="ipr",
                    fault_prob_random=1.0,
                    fault_ranks={0},
                    steps_before_fault=5,
                    fault_time_interval_sec=case["fault_time_interval_sec"],
                )

                mock_random.return_value = 0.1  # always trigger fault

                if case["mock_uniform_return"] is not None:
                    mock_uniform.return_value = case["mock_uniform_return"]

                injector = RandomErrorInjector(config, self.trainer_mock)

                # Patch stop_event.is_set to True after first check, so the loop only runs once
                with patch.object(injector.stop_event, 'is_set', side_effect=[False, True]):
                    injector.run()

                mock_sleep.assert_called_once_with(case["expected_sleep"])

                mock_sleep.reset_mock()
                mock_uniform.reset_mock()
                mock_random.reset_mock()
                mock_kill.reset_mock()

if __name__ == '__main__':
    unittest.main()
