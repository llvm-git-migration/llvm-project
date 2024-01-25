#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional, Sequence

from ._memref_ops_gen import *
from ..ir import Value, ShapedType, MemRefType, StridedLayoutAttr


def _infer_memref_subview_result_type(
    source_memref_type, static_offsets, static_sizes, static_strides
):
    source_strides, source_offset = source_memref_type.strides_and_offset
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
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if static_offsets is None:
        static_offsets = []
    if strides is None:
        strides = []
    if static_strides is None:
        static_strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    sizes = []
    S = ShapedType.get_dynamic_size()
    if offsets and static_offsets:
        assert all(s == S for s in static_offsets)
    if strides and static_strides:
        assert all(s == S for s in static_strides)
    result_type = _infer_memref_subview_result_type(
        source.type, static_offsets, static_sizes, static_strides
    )
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
