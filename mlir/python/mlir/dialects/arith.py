#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum, IntFlag

from ._arith_ops_gen import *
from ._arith_ops_gen import _Dialect
from ._arith_enum_gen import *
from array import array as _array
from typing import overload

try:
    from ..ir import *
    from ._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
        get_op_result_or_op_results as _get_op_result_or_op_results,
    )

    from typing import Any, List, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


def _isa(obj: Any, cls: type):
    try:
        cls(obj)
    except ValueError:
        return False
    return True


def _is_any_of(obj: Any, classes: List[type]):
    return any(_isa(obj, cls) for cls in classes)


def _is_integer_like_type(type: Type):
    return _is_any_of(type, [IntegerType, IndexType])


def _is_float_type(type: Type):
    return _is_any_of(type, [BF16Type, F16Type, F32Type, F64Type])


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstantOp(ConstantOp):
    """Specialization for the constant op class."""

    @overload
    def __init__(self, value: Attribute, *, loc=None, ip=None):
        ...

    @overload
    def __init__(
        self, result: Type, value: Union[int, float, _array], *, loc=None, ip=None
    ):
        ...

    def __init__(self, result, value, *, loc=None, ip=None):
        if value is None:
            assert isinstance(result, Attribute)
            super().__init__(result, loc=loc, ip=ip)
            return

        if isinstance(value, int):
            super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, float):
            super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, _array):
            if 8 * value.itemsize != result.element_type.width:
                raise ValueError(
                    f"Mismatching array element ({8 * value.itemsize}) and type ({result.element_type.width}) width."
                )
            if value.typecode in ["i", "l", "q"]:
                super().__init__(DenseIntElementsAttr.get(value, type=result))
            elif value.typecode in ["f", "d"]:
                super().__init__(DenseFPElementsAttr.get(value, type=result))
            else:
                raise ValueError(f'Unsupported typecode: "{value.typecode}".')
        else:
            super().__init__(value, loc=loc, ip=ip)

    @classmethod
    def create_index(cls, value: int, *, loc=None, ip=None):
        """Create an index-typed constant."""
        return cls(
            IndexType.get(context=_get_default_loc_context(loc)), value, loc=loc, ip=ip
        )

    @property
    def type(self):
        return self.results[0].type

    @property
    def value(self):
        return Attribute(self.operation.attributes["value"])

    @property
    def literal_value(self) -> Union[int, float]:
        if _is_integer_like_type(self.type):
            return IntegerAttr(self.value).value
        elif _is_float_type(self.type):
            return FloatAttr(self.value).value
        else:
            raise ValueError("only integer and float constants have literal values")


def constant(
    result: Type, value: Union[int, float, Attribute, _array], *, loc=None, ip=None
) -> Value:
    return _get_op_result_or_op_results(ConstantOp(result, value, loc=loc, ip=ip))


class CmpFPredicate(IntEnum):
    """allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"""

    AlwaysFalse = 0
    OEQ = 1
    OGT = 2
    OGE = 3
    OLT = 4
    OLE = 5
    ONE = 6
    ORD = 7
    UEQ = 8
    UGT = 9
    UGE = 10
    ULT = 11
    ULE = 12
    UNE = 13
    UNO = 14
    AlwaysTrue = 15

    def __str__(self):
        if self is CmpFPredicate.AlwaysFalse:
            return "false"
        if self is CmpFPredicate.OEQ:
            return "oeq"
        if self is CmpFPredicate.OGT:
            return "ogt"
        if self is CmpFPredicate.OGE:
            return "oge"
        if self is CmpFPredicate.OLT:
            return "olt"
        if self is CmpFPredicate.OLE:
            return "ole"
        if self is CmpFPredicate.ONE:
            return "one"
        if self is CmpFPredicate.ORD:
            return "ord"
        if self is CmpFPredicate.UEQ:
            return "ueq"
        if self is CmpFPredicate.UGT:
            return "ugt"
        if self is CmpFPredicate.UGE:
            return "uge"
        if self is CmpFPredicate.ULT:
            return "ult"
        if self is CmpFPredicate.ULE:
            return "ule"
        if self is CmpFPredicate.UNE:
            return "une"
        if self is CmpFPredicate.UNO:
            return "uno"
        if self is CmpFPredicate.AlwaysTrue:
            return "true"
        raise ValueError("Unknown CmpFPredicate enum entry.")


@register_attribute_builder("Arith_CmpFPredicateAttr")
def _arith_cmpfpredicateattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


class CmpIPredicate(IntEnum):
    """allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"""

    eq = 0
    ne = 1
    slt = 2
    sle = 3
    sgt = 4
    sge = 5
    ult = 6
    ule = 7
    ugt = 8
    uge = 9

    def __str__(self):
        if self is CmpIPredicate.eq:
            return "eq"
        if self is CmpIPredicate.ne:
            return "ne"
        if self is CmpIPredicate.slt:
            return "slt"
        if self is CmpIPredicate.sle:
            return "sle"
        if self is CmpIPredicate.sgt:
            return "sgt"
        if self is CmpIPredicate.sge:
            return "sge"
        if self is CmpIPredicate.ult:
            return "ult"
        if self is CmpIPredicate.ule:
            return "ule"
        if self is CmpIPredicate.ugt:
            return "ugt"
        if self is CmpIPredicate.uge:
            return "uge"
        raise ValueError("Unknown CmpIPredicate enum entry.")


@register_attribute_builder("Arith_CmpIPredicateAttr")
def _arith_cmpipredicateattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


class DenormalMode(IntEnum):
    """denormal mode arith"""

    ieee = 0
    preserve_sign = 1
    positive_zero = 2

    def __str__(self):
        if self is DenormalMode.ieee:
            return "ieee"
        if self is DenormalMode.preserve_sign:
            return "preserve_sign"
        if self is DenormalMode.positive_zero:
            return "positive_zero"
        raise ValueError("Unknown DenormalMode enum entry.")


@register_attribute_builder("Arith_DenormalMode")
def _arith_denormalmode(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class IntegerOverflowFlags(IntFlag):
    """Integer overflow arith flags"""

    none = 0
    nsw = 1
    nuw = 2

    def __iter__(self):
        return iter([case for case in type(self) if (self & case) is case])

    def __len__(self):
        return bin(self).count("1")

    def __str__(self):
        if len(self) > 1:
            return ", ".join(map(str, self))
        if self is IntegerOverflowFlags.none:
            return "none"
        if self is IntegerOverflowFlags.nsw:
            return "nsw"
        if self is IntegerOverflowFlags.nuw:
            return "nuw"
        raise ValueError("Unknown IntegerOverflowFlags enum entry.")


@register_attribute_builder("Arith_IntegerOverflowFlags")
def _arith_integeroverflowflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class RoundingMode(IntEnum):
    """Floating point rounding mode"""

    to_nearest_even = 0
    downward = 1
    upward = 2
    toward_zero = 3
    to_nearest_away = 4

    def __str__(self):
        if self is RoundingMode.to_nearest_even:
            return "to_nearest_even"
        if self is RoundingMode.downward:
            return "downward"
        if self is RoundingMode.upward:
            return "upward"
        if self is RoundingMode.toward_zero:
            return "toward_zero"
        if self is RoundingMode.to_nearest_away:
            return "to_nearest_away"
        raise ValueError("Unknown RoundingMode enum entry.")


@register_attribute_builder("Arith_RoundingModeAttr")
def _arith_roundingmodeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class AtomicRMWKind(IntEnum):
    """allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14"""

    addf = 0
    addi = 1
    assign = 2
    maximumf = 3
    maxs = 4
    maxu = 5
    minimumf = 6
    mins = 7
    minu = 8
    mulf = 9
    muli = 10
    ori = 11
    andi = 12
    maxnumf = 13
    minnumf = 14

    def __str__(self):
        if self is AtomicRMWKind.addf:
            return "addf"
        if self is AtomicRMWKind.addi:
            return "addi"
        if self is AtomicRMWKind.assign:
            return "assign"
        if self is AtomicRMWKind.maximumf:
            return "maximumf"
        if self is AtomicRMWKind.maxs:
            return "maxs"
        if self is AtomicRMWKind.maxu:
            return "maxu"
        if self is AtomicRMWKind.minimumf:
            return "minimumf"
        if self is AtomicRMWKind.mins:
            return "mins"
        if self is AtomicRMWKind.minu:
            return "minu"
        if self is AtomicRMWKind.mulf:
            return "mulf"
        if self is AtomicRMWKind.muli:
            return "muli"
        if self is AtomicRMWKind.ori:
            return "ori"
        if self is AtomicRMWKind.andi:
            return "andi"
        if self is AtomicRMWKind.maxnumf:
            return "maxnumf"
        if self is AtomicRMWKind.minnumf:
            return "minnumf"
        raise ValueError("Unknown AtomicRMWKind enum entry.")


@register_attribute_builder("AtomicRMWKindAttr")
def _atomicrmwkindattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


class FastMathFlags(IntFlag):
    """Floating point fast math flags"""

    none = 0
    reassoc = 1
    nnan = 2
    ninf = 4
    nsz = 8
    arcp = 16
    contract = 32
    afn = 64
    fast = 127

    def __iter__(self):
        return iter([case for case in type(self) if (self & case) is case])

    def __len__(self):
        return bin(self).count("1")

    def __str__(self):
        if len(self) > 1:
            return ",".join(map(str, self))
        if self is FastMathFlags.none:
            return "none"
        if self is FastMathFlags.reassoc:
            return "reassoc"
        if self is FastMathFlags.nnan:
            return "nnan"
        if self is FastMathFlags.ninf:
            return "ninf"
        if self is FastMathFlags.nsz:
            return "nsz"
        if self is FastMathFlags.arcp:
            return "arcp"
        if self is FastMathFlags.contract:
            return "contract"
        if self is FastMathFlags.afn:
            return "afn"
        if self is FastMathFlags.fast:
            return "fast"
        raise ValueError("Unknown FastMathFlags enum entry.")


@register_attribute_builder("FastMathFlags")
def _fastmathflags(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
