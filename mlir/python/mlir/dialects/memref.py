#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional

from ._memref_ops_gen import *
from .arith import ConstantOp, _is_integer_like_type
from .transform.structured import _dispatch_mixed_values, MixedValues
from ..ir import Value, MemRefType, StridedLayoutAttr, ShapedType


def _infer_memref_subview_result_type(
    source_memref_type, static_offsets, static_sizes, static_strides
):
    source_strides, source_offset = source_memref_type.strides_and_offset
    assert all(
        all(
            (isinstance(i, int) and not ShapedType.is_dynamic_size(i))
            or (isinstance(i, Value) and isinstance(i.owner.opview, ConstantOp))
            and _is_integer_like_type(i.type)
            for i in s
        )
        for s in [
            static_offsets,
            static_sizes,
            static_strides,
            source_strides,
            [source_offset],
        ]
    ), f"Only inferring from python or mlir integer constant is supported"
    for s in [static_offsets, static_sizes, static_strides]:
        for idx, i in enumerate(s):
            if isinstance(i, Value):
                s[idx] = i.owner.opview.literal_value

    target_offset = source_offset
    for static_offset, target_stride in zip(static_offsets, source_strides):
        target_offset += static_offset * target_stride

    target_strides = []
    for source_stride, static_stride in zip(source_strides, static_strides):
        target_strides.append(source_stride * static_stride)

    layout = StridedLayoutAttr.get(target_offset, target_strides)
    return MemRefType.get(
        static_sizes,
        source_memref_type.element_type,
        layout,
        source_memref_type.memory_space,
    )


_generated_subview = subview


def subview(
    source: Value,
    offsets: MixedValues,
    sizes: MixedValues,
    strides: MixedValues,
    *,
    result_type: Optional[MemRefType] = None,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if sizes is None:
        sizes = []
    if strides is None:
        strides = []

    source_strides, source_offset = source.type.strides_and_offset
    if all(
        all(
            (isinstance(i, int) and not ShapedType.is_dynamic_size(i))
            or (isinstance(i, Value) and isinstance(i.owner.opview, ConstantOp))
            for i in s
        )
        for s in [offsets, sizes, strides, source_strides, [source_offset]]
    ):
        result_type = _infer_memref_subview_result_type(
            source.type, offsets, sizes, strides
        )
    else:
        assert (
            result_type is not None
        ), "mixed static/dynamic offset/sizes/strides requires explicit result type"

    offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(offsets)
    sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
    strides, _packed_strides, static_strides = _dispatch_mixed_values(strides)

    return _generated_subview(
        result_type,
        source,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )
