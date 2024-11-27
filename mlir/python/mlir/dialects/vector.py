#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._vector_ops_gen import *
from ._vector_enum_gen import *


class CombiningKind(IntEnum):
    """Kind of combining function for contractions and reductions"""

    ADD = 0
    MUL = 1
    MINUI = 2
    MINSI = 3
    MINNUMF = 4
    MAXUI = 5
    MAXSI = 6
    MAXNUMF = 7
    AND = 8
    OR = 9
    XOR = 10
    MAXIMUMF = 12
    MINIMUMF = 11

    def __str__(self):
        if self is CombiningKind.ADD:
            return "add"
        if self is CombiningKind.MUL:
            return "mul"
        if self is CombiningKind.MINUI:
            return "minui"
        if self is CombiningKind.MINSI:
            return "minsi"
        if self is CombiningKind.MINNUMF:
            return "minnumf"
        if self is CombiningKind.MAXUI:
            return "maxui"
        if self is CombiningKind.MAXSI:
            return "maxsi"
        if self is CombiningKind.MAXNUMF:
            return "maxnumf"
        if self is CombiningKind.AND:
            return "and"
        if self is CombiningKind.OR:
            return "or"
        if self is CombiningKind.XOR:
            return "xor"
        if self is CombiningKind.MAXIMUMF:
            return "maximumf"
        if self is CombiningKind.MINIMUMF:
            return "minimumf"
        raise ValueError("Unknown CombiningKind enum entry.")


@register_attribute_builder("CombiningKind")
def _combiningkind(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class PrintPunctuation(IntEnum):
    """Punctuation for separating vectors or vector elements"""

    NoPunctuation = 0
    NewLine = 1
    Comma = 2
    Open = 3
    Close = 4

    def __str__(self):
        if self is PrintPunctuation.NoPunctuation:
            return "no_punctuation"
        if self is PrintPunctuation.NewLine:
            return "newline"
        if self is PrintPunctuation.Comma:
            return "comma"
        if self is PrintPunctuation.Open:
            return "open"
        if self is PrintPunctuation.Close:
            return "close"
        raise ValueError("Unknown PrintPunctuation enum entry.")


@register_attribute_builder("PrintPunctuation")
def _printpunctuation(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class IteratorType(IntEnum):
    """Iterator type"""

    parallel = 0
    reduction = 1

    def __str__(self):
        if self is IteratorType.parallel:
            return "parallel"
        if self is IteratorType.reduction:
            return "reduction"
        raise ValueError("Unknown IteratorType enum entry.")


@register_attribute_builder("Vector_IteratorType")
def _vector_iteratortype(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
