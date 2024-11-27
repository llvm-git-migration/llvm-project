#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._bufferization_ops_gen import *


class LayoutMapOption(IntEnum):
    """option for map layout"""

    InferLayoutMap = 0
    IdentityLayoutMap = 1
    FullyDynamicLayoutMap = 2

    def __str__(self):
        if self is LayoutMapOption.InferLayoutMap:
            return "InferLayoutMap"
        if self is LayoutMapOption.IdentityLayoutMap:
            return "IdentityLayoutMap"
        if self is LayoutMapOption.FullyDynamicLayoutMap:
            return "FullyDynamicLayoutMap"
        raise ValueError("Unknown LayoutMapOption enum entry.")


@register_attribute_builder("LayoutMapOption")
def _layoutmapoption(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
