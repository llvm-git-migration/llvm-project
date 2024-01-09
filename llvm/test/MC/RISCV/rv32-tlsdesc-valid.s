# RUN: llvm-mc -filetype=obj -triple riscv32 < %s --defsym BIT32=1  | llvm-objdump -d -M no-aliases - | FileCheck %s --check-prefix=INST
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s | llvm-objdump -d -M no-aliases - | FileCheck %s

start:                                  # @start
# %bb.0:                                # %entry
.Ltlsdesc_hi0:
	auipc a0, %tlsdesc_hi(a-4)
	# INST: auipc a0, 0x0
	auipc	a0, %tlsdesc_hi(unspecified)
	# INST: auipc a0, 0x0
.ifdef BIT32
	lw	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	# INST: lw a1, 0x0(a0)
.else
	ld	a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
	#INST: ld a1, 0x0(a0)
.endif
	addi	a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
	# INST: addi a0, a0, 0x0
	jalr	t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
	# INST: jalr t0, 0x0(a1)
	add	a0, a0, tp
	# INST: add a0, a0, tp
	ret
