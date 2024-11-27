#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from enum import IntEnum, IntFlag

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._amdgpu_ops_gen import *
from ._amdgpu_enum_gen import *


class DPPPerm(IntEnum):
    """The possible permutations for a DPP operation"""

    quad_perm = 0
    row_shl = 1
    row_shr = 2
    row_ror = 3
    wave_shl = 4
    wave_shr = 5
    wave_ror = 6
    wave_rol = 7
    row_mirror = 8
    row_half_mirror = 9
    row_bcast_15 = 10
    row_bcast_31 = 11

    def __str__(self):
        if self is DPPPerm.quad_perm:
            return "quad_perm"
        if self is DPPPerm.row_shl:
            return "row_shl"
        if self is DPPPerm.row_shr:
            return "row_shr"
        if self is DPPPerm.row_ror:
            return "row_ror"
        if self is DPPPerm.wave_shl:
            return "wave_shl"
        if self is DPPPerm.wave_shr:
            return "wave_shr"
        if self is DPPPerm.wave_ror:
            return "wave_ror"
        if self is DPPPerm.wave_rol:
            return "wave_rol"
        if self is DPPPerm.row_mirror:
            return "row_mirror"
        if self is DPPPerm.row_half_mirror:
            return "row_half_mirror"
        if self is DPPPerm.row_bcast_15:
            return "row_bcast_15"
        if self is DPPPerm.row_bcast_31:
            return "row_bcast_31"
        raise ValueError("Unknown DPPPerm enum entry.")


@register_attribute_builder("AMDGPU_DPPPerm")
def _amdgpu_dppperm(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class MFMAPermB(IntEnum):
    """The possible permutations of the lanes storing B available in an MFMA"""

    none = 0
    bcast_first_32 = 1
    bcast_second_32 = 2
    rotate_16_right = 3
    bcast_first_16 = 4
    bcast_second_16 = 5
    bcast_third_16 = 6
    bcast_fourth_16 = 7

    def __str__(self):
        if self is MFMAPermB.none:
            return "none"
        if self is MFMAPermB.bcast_first_32:
            return "bcast_first_32"
        if self is MFMAPermB.bcast_second_32:
            return "bcast_second_32"
        if self is MFMAPermB.rotate_16_right:
            return "rotate_16_right"
        if self is MFMAPermB.bcast_first_16:
            return "bcast_first_16"
        if self is MFMAPermB.bcast_second_16:
            return "bcast_second_16"
        if self is MFMAPermB.bcast_third_16:
            return "bcast_third_16"
        if self is MFMAPermB.bcast_fourth_16:
            return "bcast_fourth_16"
        raise ValueError("Unknown MFMAPermB enum entry.")


@register_attribute_builder("AMDGPU_MFMAPermB")
def _amdgpu_mfmapermb(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


class sched_barrier_opt_enum(IntFlag):
    """The possible options for scheduling barriers"""

    none = 0
    non_mem_non_sideffect = 1
    valu = 2
    salu = 4
    mfma_wmma = 8
    all_vmem = 16
    vmem_read = 32
    vmem_write = 64
    all_ds = 128
    ds_read = 256
    ds_write = 512
    transcendental = 1024

    def __iter__(self):
        return iter([case for case in type(self) if (self & case) is case])

    def __len__(self):
        return bin(self).count("1")

    def __str__(self):
        if len(self) > 1:
            return "|".join(map(str, self))
        if self is sched_barrier_opt_enum.none:
            return "none"
        if self is sched_barrier_opt_enum.non_mem_non_sideffect:
            return "non_mem_non_sideffect"
        if self is sched_barrier_opt_enum.valu:
            return "valu"
        if self is sched_barrier_opt_enum.salu:
            return "salu"
        if self is sched_barrier_opt_enum.mfma_wmma:
            return "mfma_wmma"
        if self is sched_barrier_opt_enum.all_vmem:
            return "all_vmem"
        if self is sched_barrier_opt_enum.vmem_read:
            return "vmem_read"
        if self is sched_barrier_opt_enum.vmem_write:
            return "vmem_write"
        if self is sched_barrier_opt_enum.all_ds:
            return "all_ds"
        if self is sched_barrier_opt_enum.ds_read:
            return "ds_read"
        if self is sched_barrier_opt_enum.ds_write:
            return "ds_write"
        if self is sched_barrier_opt_enum.transcendental:
            return "transcendental"
        raise ValueError("Unknown sched_barrier_opt_enum enum entry.")


@register_attribute_builder("AMDGPU_SchedBarrierOpOpt")
def _amdgpu_schedbarrieropopt(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
