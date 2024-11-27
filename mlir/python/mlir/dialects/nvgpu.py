#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._nvgpu_ops_gen import *
from ._nvgpu_enum_gen import *
from .._mlir_libs._mlirDialectsNVGPU import *


class RcpRoundingMode(IntEnum):
    """Rounding mode of rcp"""

    APPROX = 0
    RN = 1
    RZ = 2
    RM = 3
    RP = 4

    def __str__(self):
        if self is RcpRoundingMode.APPROX:
            return "approx"
        if self is RcpRoundingMode.RN:
            return "rn"
        if self is RcpRoundingMode.RZ:
            return "rz"
        if self is RcpRoundingMode.RM:
            return "rm"
        if self is RcpRoundingMode.RP:
            return "rp"
        raise ValueError("Unknown RcpRoundingMode enum entry.")


@register_attribute_builder("RcpRoundingMode")
def _rcproundingmode(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class TensorMapInterleaveKind(IntEnum):
    """Tensor map interleave layout type"""

    INTERLEAVE_NONE = 0
    INTERLEAVE_16B = 1
    INTERLEAVE_32B = 2

    def __str__(self):
        if self is TensorMapInterleaveKind.INTERLEAVE_NONE:
            return "none"
        if self is TensorMapInterleaveKind.INTERLEAVE_16B:
            return "interleave_16b"
        if self is TensorMapInterleaveKind.INTERLEAVE_32B:
            return "interleave_32b"
        raise ValueError("Unknown TensorMapInterleaveKind enum entry.")


@register_attribute_builder("TensorMapInterleaveKind")
def _tensormapinterleavekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class TensorMapL2PromoKind(IntEnum):
    """Tensor map L2 promotion type"""

    L2PROMO_NONE = 0
    L2PROMO_64B = 1
    L2PROMO_128B = 2
    L2PROMO_256B = 3

    def __str__(self):
        if self is TensorMapL2PromoKind.L2PROMO_NONE:
            return "none"
        if self is TensorMapL2PromoKind.L2PROMO_64B:
            return "l2promo_64b"
        if self is TensorMapL2PromoKind.L2PROMO_128B:
            return "l2promo_128b"
        if self is TensorMapL2PromoKind.L2PROMO_256B:
            return "l2promo_256b"
        raise ValueError("Unknown TensorMapL2PromoKind enum entry.")


@register_attribute_builder("TensorMapL2PromoKind")
def _tensormapl2promokind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class TensorMapOOBKind(IntEnum):
    """Tensor map out-of-bounds fill type"""

    OOB_ZERO = 0
    OOB_NAN = 1

    def __str__(self):
        if self is TensorMapOOBKind.OOB_ZERO:
            return "zero"
        if self is TensorMapOOBKind.OOB_NAN:
            return "nan"
        raise ValueError("Unknown TensorMapOOBKind enum entry.")


@register_attribute_builder("TensorMapOOBKind")
def _tensormapoobkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class TensorMapSwizzleKind(IntEnum):
    """Tensor map swizzling mode of shared memory banks"""

    SWIZZLE_NONE = 0
    SWIZZLE_32B = 1
    SWIZZLE_64B = 2
    SWIZZLE_128B = 3

    def __str__(self):
        if self is TensorMapSwizzleKind.SWIZZLE_NONE:
            return "none"
        if self is TensorMapSwizzleKind.SWIZZLE_32B:
            return "swizzle_32b"
        if self is TensorMapSwizzleKind.SWIZZLE_64B:
            return "swizzle_64b"
        if self is TensorMapSwizzleKind.SWIZZLE_128B:
            return "swizzle_128b"
        raise ValueError("Unknown TensorMapSwizzleKind enum entry.")


@register_attribute_builder("TensorMapSwizzleKind")
def _tensormapswizzlekind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
