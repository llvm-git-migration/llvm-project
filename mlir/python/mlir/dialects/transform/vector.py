#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ...ir import IntegerAttr, IntegerType, register_attribute_builder
from .._vector_transform_ops_gen import *


class VectorContractLowering(IntEnum):
    """control the lowering of `vector.contract` operations."""

    Dot = 0
    Matmul = 1
    OuterProduct = 2
    ParallelArith = 3

    def __str__(self):
        if self is VectorContractLowering.Dot:
            return "dot"
        if self is VectorContractLowering.Matmul:
            return "matmulintrinsics"
        if self is VectorContractLowering.OuterProduct:
            return "outerproduct"
        if self is VectorContractLowering.ParallelArith:
            return "parallelarith"
        raise ValueError("Unknown VectorContractLowering enum entry.")


@register_attribute_builder("VectorContractLoweringAttr")
def _vectorcontractloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class VectorMultiReductionLowering(IntEnum):
    """control the lowering of `vector.multi_reduction`."""

    InnerParallel = 0
    InnerReduction = 1

    def __str__(self):
        if self is VectorMultiReductionLowering.InnerParallel:
            return "innerparallel"
        if self is VectorMultiReductionLowering.InnerReduction:
            return "innerreduction"
        raise ValueError("Unknown VectorMultiReductionLowering enum entry.")


@register_attribute_builder("VectorMultiReductionLoweringAttr")
def _vectormultireductionloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class VectorTransferSplit(IntEnum):
    """control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants."""

    None_ = 0
    VectorTransfer = 1
    LinalgCopy = 2
    ForceInBounds = 3

    def __str__(self):
        if self is VectorTransferSplit.None_:
            return "none"
        if self is VectorTransferSplit.VectorTransfer:
            return "vector-transfer"
        if self is VectorTransferSplit.LinalgCopy:
            return "linalg-copy"
        if self is VectorTransferSplit.ForceInBounds:
            return "force-in-bounds"
        raise ValueError("Unknown VectorTransferSplit enum entry.")


@register_attribute_builder("VectorTransferSplitAttr")
def _vectortransfersplitattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class VectorTransposeLowering(IntEnum):
    """control the lowering of `vector.transpose` operations."""

    EltWise = 0
    Flat = 1
    Shuffle1D = 2
    Shuffle16x16 = 3

    def __str__(self):
        if self is VectorTransposeLowering.EltWise:
            return "eltwise"
        if self is VectorTransposeLowering.Flat:
            return "flat_transpose"
        if self is VectorTransposeLowering.Shuffle1D:
            return "shuffle_1d"
        if self is VectorTransposeLowering.Shuffle16x16:
            return "shuffle_16x16"
        raise ValueError("Unknown VectorTransposeLowering enum entry.")


@register_attribute_builder("VectorTransposeLoweringAttr")
def _vectortransposeloweringattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
