#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._index_ops_gen import *
from ._index_enum_gen import *


class IndexCmpPredicate(IntEnum):
    """index comparison predicate kind"""

    EQ = 0
    NE = 1
    SLT = 2
    SLE = 3
    SGT = 4
    SGE = 5
    ULT = 6
    ULE = 7
    UGT = 8
    UGE = 9

    def __str__(self):
        if self is IndexCmpPredicate.EQ:
            return "eq"
        if self is IndexCmpPredicate.NE:
            return "ne"
        if self is IndexCmpPredicate.SLT:
            return "slt"
        if self is IndexCmpPredicate.SLE:
            return "sle"
        if self is IndexCmpPredicate.SGT:
            return "sgt"
        if self is IndexCmpPredicate.SGE:
            return "sge"
        if self is IndexCmpPredicate.ULT:
            return "ult"
        if self is IndexCmpPredicate.ULE:
            return "ule"
        if self is IndexCmpPredicate.UGT:
            return "ugt"
        if self is IndexCmpPredicate.UGE:
            return "uge"
        raise ValueError("Unknown IndexCmpPredicate enum entry.")


@register_attribute_builder("IndexCmpPredicate")
def _indexcmppredicate(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
