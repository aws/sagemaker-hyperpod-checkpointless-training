"""Original Copyright 2024 NVIDIA CORPORATION & AFFILIATES under the Apache License, Version 2.0"""
"""Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved"""

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
