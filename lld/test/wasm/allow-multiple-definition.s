# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/allow-multiple-definition.s -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefix=DEF1 %s
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck --check-prefix=DEF2 %s
# RUN: not wasm-ld -o %t.wasm %t1 %t2 2>&1 | FileCheck --check-prefix=DUP1 %s
# RUN: not wasm-ld -o %t.wasm %t2 %t1 2>&1 | FileCheck --check-prefix=DUP2 %s
# RUN: wasm-ld --allow-multiple-definition %t1 %t2 -o %t3
# RUN: llvm-objdump --no-print-imm-hex -d %t3 | FileCheck --check-prefix=RES12 %s
# RUN: wasm-ld --allow-multiple-definition %t2 %t1 -o %t4
# RUN: llvm-objdump --no-print-imm-hex -d %t4 | FileCheck --check-prefix=RES21 %s


# DUP1: duplicate symbol: foo
# DUP2: duplicate symbol: foo

# DEF1: i32.const 0
# DEF2: i32.const 1

# RES12: i32.const 0
# RES21: i32.const 1

# inputs contain different constants for function foo return.
# Tests below checks that order of files in command line
# affects on what symbol will be used.
# If flag allow-multiple-definition is enabled the first
# meet symbol should be used.

  .hidden foo
  .globl  foo
foo:
  .functype foo () -> (i32)
  i32.const 0
  end_function

	.globl _start
_start:
	.functype	_start () -> (i32)
	call foo
	end_function