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

# Portions taken from NVIDIA nvidia-resiliency-ext, Copyright Nvidia Corporation

import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar('T')


def find_common_ancestor(*instances):
    common_mro = set(type(instances[0]).mro())

    for instance in instances[1:]:
        common_mro &= set(type(instance).mro())

    if common_mro:
        mro_list = type(instances[0]).mro()
        common_ancestor = [cls for cls in mro_list if cls in common_mro]
        return common_ancestor[0]
    else:
        return None


class Compose:
    def __new__(cls, *instances: Callable[[T], T]):

        common_ancestor = find_common_ancestor(*instances)
        DynamicCompose = type(
            'DynamicCompose',
            (Compose, common_ancestor),
            {
                'instances': instances,
                '__new__': object.__new__,
            },
        )
        return DynamicCompose()

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        ex = []
        rc = []
        for instance in reversed(self.instances):
            try:
                rc.append(instance(*args, **kwargs))
            except Exception as e:
                ex.append(e)

        if ex:
            msg = "errors: {ex}"
            raise Exception(msg)

        return rc
