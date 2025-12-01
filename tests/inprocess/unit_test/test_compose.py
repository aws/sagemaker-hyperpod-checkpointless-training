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
from unittest.mock import Mock
from hyperpod_checkpointless_training.inprocess.compose import Compose, find_common_ancestor


class TestFindCommonAncestor:
    def test_single_instance(self):
        mock = Mock()
        result = find_common_ancestor(mock)
        assert result == type(mock)

    def test_multiple_instances_different_types(self):
        class A: pass
        class B(A): pass

        a = A()
        b = B()
        result = find_common_ancestor(a, b)
        assert result == A


class TestCompose:
    def test_init_with_single_callable(self):
        mock = Mock()
        compose = Compose(mock)
        assert compose.instances == (mock,)

    def test_init_with_multiple_callables(self):
        mock1 = Mock()
        mock2 = Mock()
        compose = Compose(mock1, mock2)
        assert compose.instances == (mock1, mock2)

    def test_call_success_all_instances(self):
        mock1 = Mock(return_value="result1")
        mock2 = Mock(return_value="result2")
        compose = Compose(mock1, mock2)

        result = compose("arg1", kwarg1="value1")

        assert result == ["result2", "result1"]
        mock1.assert_called_once_with("arg1", kwarg1="value1")
        mock2.assert_called_once_with("arg1", kwarg1="value1")

    def test_call_with_exception(self):
        mock1 = Mock(side_effect=ValueError("error1"))
        mock2 = Mock(return_value="result2")
        compose = Compose(mock1, mock2)

        with pytest.raises(Exception, match="errors:"):
            compose("arg1")

    def test_call_multiple_exceptions(self):
        mock1 = Mock(side_effect=ValueError("error1"))
        mock2 = Mock(side_effect=RuntimeError("error2"))
        compose = Compose(mock1, mock2)

        with pytest.raises(Exception, match="errors:"):
            compose("arg1")

    def test_call_no_args(self):
        mock1 = Mock(return_value="result1")
        mock2 = Mock(return_value="result2")
        compose = Compose(mock1, mock2)

        result = compose()

        assert result == ["result2", "result1"]
        mock1.assert_called_once_with()
        mock2.assert_called_once_with()
