# REQUIRES: x86
# RUN: split-file %s %t.dir

# RUN: llvm-dlltool -m i386:x86-64 -d %t.dir/lib.def -D lib.dll -l %t.dir/lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/main.s -filetype=obj -o %t.dir/main.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/func.s -filetype=obj -o %t.dir/func.obj
# RUN: lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/lib.lib 2>&1 | FileCheck %s --check-prefix=WARN

# RUN: lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/func.obj %t.dir/lib.lib 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty

# WARN: warning: output image has pseudo relocations, but function _pei386_runtime_relocator missing; the relocations might not get fixed at runtime

# NOWARN-NOT: warning

#--- main.s
    .global main
    .text
main:
    ret

    .data
    .long 1
    .quad variable
    .long 2

#--- func.s
    .global _pei386_runtime_relocator
    .text
_pei386_runtime_relocator:
    ret

#--- lib.def
EXPORTS
variable DATA

