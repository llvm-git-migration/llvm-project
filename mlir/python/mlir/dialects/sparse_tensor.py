#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._sparse_tensor_ops_gen import *
from ._sparse_tensor_enum_gen import *
from .._mlir_libs._mlirDialectsSparseTensor import *


class CrdTransDirectionKind(IntEnum):
    """sparse tensor coordinate translation direction"""

    dim2lvl = 0
    lvl2dim = 1

    def __str__(self):
        if self is CrdTransDirectionKind.dim2lvl:
            return "dim_to_lvl"
        if self is CrdTransDirectionKind.lvl2dim:
            return "lvl_to_dim"
        raise ValueError("Unknown CrdTransDirectionKind enum entry.")


@register_attribute_builder("SparseTensorCrdTransDirectionEnum")
def _sparsetensorcrdtransdirectionenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class SparseTensorSortKind(IntEnum):
    """sparse tensor sort algorithm"""

    HybridQuickSort = 0
    InsertionSortStable = 1
    QuickSort = 2
    HeapSort = 3

    def __str__(self):
        if self is SparseTensorSortKind.HybridQuickSort:
            return "hybrid_quick_sort"
        if self is SparseTensorSortKind.InsertionSortStable:
            return "insertion_sort_stable"
        if self is SparseTensorSortKind.QuickSort:
            return "quick_sort"
        if self is SparseTensorSortKind.HeapSort:
            return "heap_sort"
        raise ValueError("Unknown SparseTensorSortKind enum entry.")


@register_attribute_builder("SparseTensorSortKindEnum")
def _sparsetensorsortkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class StorageSpecifierKind(IntEnum):
    """sparse tensor storage specifier kind"""

    LvlSize = 0
    PosMemSize = 1
    CrdMemSize = 2
    ValMemSize = 3
    DimOffset = 4
    DimStride = 5

    def __str__(self):
        if self is StorageSpecifierKind.LvlSize:
            return "lvl_sz"
        if self is StorageSpecifierKind.PosMemSize:
            return "pos_mem_sz"
        if self is StorageSpecifierKind.CrdMemSize:
            return "crd_mem_sz"
        if self is StorageSpecifierKind.ValMemSize:
            return "val_mem_sz"
        if self is StorageSpecifierKind.DimOffset:
            return "dim_offset"
        if self is StorageSpecifierKind.DimStride:
            return "dim_stride"
        raise ValueError("Unknown StorageSpecifierKind enum entry.")


@register_attribute_builder("SparseTensorStorageSpecifierKindEnum")
def _sparsetensorstoragespecifierkindenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
