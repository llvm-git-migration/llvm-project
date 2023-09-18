# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.obj
# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig.obj

# RUN: lld-link -machine:arm64ec -dll -noentry -out:%t1.dll %t.obj %t-loadconfig.obj "-alternatename:#func=altsym"

# RUN: llvm-objdump -d %t1.dll | FileCheck --check-prefix=DISASM %s
# DISASM:      0000000180001000 <.text>:
# DISASM-NEXT: 180001000: 52800020     mov     w0, #0x1                // =1
# DISASM-NEXT: 180001004: d65f03c0     ret
# DISASM-NOT: .thnk

# RUN: llvm-readobj --hex-dump=.test %t1.dll | FileCheck --check-prefix=TESTSEC %s
# TESTSEC: 0x180004000 00100000 00100000

# RUN: lld-link -machine:arm64ec -dll -noentry -out:%t2.dll %t.obj %t-loadconfig.obj -alternatename:func=altsym

# RUN: llvm-objdump -d %t2.dll | FileCheck --check-prefix=DISASM2 %s
# DISASM2:      Disassembly of section .text:
# DISASM2-EMPTY:
# DISASM2-NEXT: 0000000180001000 <.text>:
# DISASM2-NEXT: 180001000: 52800020     mov     w0, #0x1                // =1
# DISASM2-NEXT: 180001004: d65f03c0     ret
# DISASM2-EMPTY:
# DISASM2-NEXT: Disassembly of section .thnk:
# DISASM2-EMPTY:
# DISASM2-NEXT: 0000000180005000 <.thnk>:
# DISASM2-NEXT: 180005000: 52800040     mov     w0, #0x2                // =2
# DISASM2-NEXT: 180005004: d65f03c0     ret

# RUN: llvm-readobj --hex-dump=.test %t2.dll | FileCheck --check-prefix=TESTSEC2 %s
# TESTSEC2: 0x180004000 00100000 00500000

        .weak_anti_dep func
.set func, "#func"
        .weak_anti_dep "#func"
.set "#func", thunksym

        .section .test, "r"
        .rva func
        .rva "#func"

        .section .thnk,"xr",discard,thunksym
thunksym:
        mov w0, #2
        ret

        .section .text,"xr",discard,altsym
        .globl altsym
altsym:
        mov w0, #1
        ret
