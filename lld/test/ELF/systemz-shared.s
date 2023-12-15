// REQUIRES: systemz
// RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=%t2.so -o %t2.so
// RUN: llvm-readelf -S %t2.so | FileCheck --check-prefix=SO %s

// RUN: ld.lld -dynamic-linker /lib/ld64.so.1 -rpath foo -rpath bar --export-dynamic %t.o %t2.so -o %t
// RUN: llvm-readelf -S -s -l --dynamic %t | FileCheck %s

// RUN: ld.lld %t.o %t2.so %t2.so -o %t2
// RUN: llvm-readelf --dyn-syms %t2 | FileCheck --check-prefix=DONT_EXPORT %s

// Make sure .symtab is properly aligned.
// SO: .symtab SYMTAB 0000000000000000 {{[^ ]*}} {{[^ ]*}} {{[^ ]*}} {{[^ ]*}} {{[^ ]*}} 8

// CHECK: Section Headers:
// CHECK-NEXT:  [Nr] Name              Type            Address              Off    Size   ES Flg Lk Inf Al
// CHECK-NEXT:  [ 0]                   NULL            0000000000000000     000000 000000 00      0   0  0
// CHECK-NEXT:  [ 1] .interp           PROGBITS        0000000001000200     000200 00000f 00   A  0   0  1
// CHECK-NEXT:  [ 2] .dynsym           DYNSYM          0000000001000210     000210 000060 18   A  5   1  8
// CHECK-NEXT:  [ 3] .gnu.hash         GNU_HASH        0000000001000270     000270 000020 00   A  2   0  8
// CHECK-NEXT:  [ 4] .hash             HASH            0000000001000290     000290 000028 04   A  2   0  4
// CHECK-NEXT:  [ 5] .dynstr           STRTAB          00000000010002b8     0002b8 {{.*}} 00   A  0   0  1
// CHECK-NEXT:  [ 6] .rela.dyn         RELA            {{0*}}[[RELADYN:.*]] {{.*}} 000030 18   A  2   0  8
// CHECK-NEXT:  [ 7] .text             PROGBITS        {{0*}}[[TEXT:.*]]    {{.*}} 00000c 00  AX  0   0  4
// CHECK-NEXT:  [ 8] .dynamic          DYNAMIC         {{0*}}[[DYNAMIC:.*]] {{.*}} 0000d0 10  WA  5   0  8
// CHECK-NEXT:  [ 9] .got              PROGBITS        {{.*}}               {{.*}} 000028 00  WA  0   0  8
// CHECK-NEXT:  [10] .relro_padding    NOBITS          {{.*}}               {{.*}} {{.*}} 00  WA  0   0  1
// CHECK-NEXT:  [11] .comment          PROGBITS        0000000000000000     {{.*}} 000008 01  MS  0   0  1
// CHECK-NEXT:  [12] .symtab           SYMTAB          0000000000000000     {{.*}} {{.*}} 18     14   2  8
// CHECK-NEXT:  [13] .shstrtab         STRTAB          0000000000000000     {{.*}} {{.*}} 00      0   0  1
// CHECK-NEXT:  [14] .strtab           STRTAB          0000000000000000     {{.*}} {{.*}} 00      0   0  1

// CHECK: Program Headers:
// CHECK-NEXT:  Type           Offset   VirtAddr            PhysAddr            FileSiz  MemSiz   Flg Align
// CHECK-NEXT:  PHDR           0x000040 0x0000000001000040  0x0000000001000040  0x0001c0 0x0001c0 R   0x8
// CHECK-NEXT:  INTERP         0x000200 0x0000000001000200  0x0000000001000200  0x00000f 0x00000f R   0x1
// CHECK-NEXT:      [Requesting program interpreter: /lib/ld64.so.1]
// CHECK-NEXT:  LOAD           0x000000 0x0000000001000000  0x0000000001000000  0x{{.*}} 0x{{.*}} R   0x1000
// CHECK-NEXT:  LOAD           0x{{.*}} 0x{{0*}}[[TEXT]]    0x{{0*}}[[TEXT]]    0x00000c 0x00000c R E 0x1000
// CHECK-NEXT:  LOAD           0x{{.*}} 0x{{0*}}[[DYNAMIC]] 0x{{0*}}[[DYNAMIC]] 0x{{.*}} 0x{{.*}} RW  0x1000
// CHECK-NEXT:  DYNAMIC        0x{{.*}} 0x{{0*}}[[DYNAMIC]] 0x{{0*}}[[DYNAMIC]] 0x0000d0 0x0000d0 RW  0x8
// CHECK-NEXT:  GNU_RELRO      0x{{.*}} 0x{{0*}}[[DYNAMIC]] 0x{{0*}}[[DYNAMIC]] 0x{{.*}} 0x{{.*}} R   0x1
// CHECK-NEXT:  GNU_STACK      0x000000 0x0000000000000000  0x0000000000000000  0x000000 0x000000 RW  0x0

// CHECK: Dynamic section at offset {{.*}} contains 13 entries:
// CHECK-NEXT:   Tag                Type       Name/Value
// CHECK-NEXT:   0x000000000000001d (RUNPATH)  Library runpath: [foo:bar]
// CHECK-NEXT:   0x0000000000000001 (NEEDED)   Shared library: [{{.*}}2.so]
// CHECK-NEXT:   0x0000000000000015 (DEBUG)    0x0
// CHECK-NEXT:   0x0000000000000007 (RELA)     0x[[RELADYN]]
// CHECK-NEXT:   0x0000000000000008 (RELASZ)   48 (bytes)
// CHECK-NEXT:   0x0000000000000009 (RELAENT)  24 (bytes)
// CHECK-NEXT:   0x0000000000000006 (SYMTAB)   0x1000210
// CHECK-NEXT:   0x000000000000000b (SYMENT)   24 (bytes)
// CHECK-NEXT:   0x0000000000000005 (STRTAB)   0x10002b8
// CHECK-NEXT:   0x000000000000000a (STRSZ)    {{.*}} (bytes)
// CHECK-NEXT:   0x000000006ffffef5 (GNU_HASH) 0x
// CHECK-NEXT:   0x0000000000000004 (HASH)     0x
// CHECK-NEXT:   0x0000000000000000 (NULL)     0x0

// CHECK:      Symbol table '.dynsym' contains 4 entries:
// CHECK-NEXT:   Num:    Value          Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK-NEXT:     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND bar
// CHECK-NEXT:     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND zed
// CHECK-NEXT:     3: {{.*}}               0 NOTYPE  GLOBAL DEFAULT     7 _start

// CHECK:      Symbol table '.symtab' contains 5 entries:
// CHECK-NEXT:   Num:    Value          Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK-NEXT:     1: {{0*}}[[DYNAMIC]]    0 NOTYPE  LOCAL  HIDDEN      8 _DYNAMIC
// CHECK-NEXT:     2: {{.*}}               0 NOTYPE  GLOBAL DEFAULT     7 _start
// CHECK-NEXT:     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND bar
// CHECK-NEXT:     4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND zed

// DONT_EXPORT:      Symbol table '.dynsym' contains 3 entries:
// DONT_EXPORT-NEXT:   Num:    Value          Size Type    Bind   Vis     Ndx Name
// DONT_EXPORT-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT UND
// DONT_EXPORT-NEXT:     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT UND bar
// DONT_EXPORT-NEXT:     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT UND zed

.global _start
_start:
	lgrl  %r1,bar@GOT
	lgrl  %r2,zed@GOT
