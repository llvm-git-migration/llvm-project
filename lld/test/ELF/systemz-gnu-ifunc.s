// REQUIRES: systemz
// RUN: llvm-mc -filetype=obj -triple=s390x-none-linux-gnu %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readelf --section-headers --relocations --symbols %tout | FileCheck %s

// CHECK:      There are 9 section headers
// CHECK:      Section Headers:
// CHECK-NEXT:  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
// CHECK-NEXT:  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
// CHECK-NEXT:  [ 1] .rela.dyn         RELA            0000000001000158 000158 000030 18  AI  0   4  8
// CHECK-NEXT:  [ 2] .text             PROGBITS        0000000001001188 000188 00001c 00  AX  0   0  4
// CHECK-NEXT:  [ 3] .iplt             PROGBITS        00000000010011b0 0001b0 000040 00  AX  0   0 16
// CHECK-NEXT:  [ 4] .got.plt          PROGBITS        00000000010021f0 0001f0 000010 00  WA  0   0  8
// CHECK-NEXT:  [ 5] .comment          PROGBITS        0000000000000000 000200 000008 01  MS  0   0  1
// CHECK-NEXT:  [ 6] .symtab           SYMTAB          0000000000000000 000208 000090 18      8   3  8
// CHECK-NEXT:  [ 7] .shstrtab         STRTAB          0000000000000000 000298 000043 00      0   0  1
// CHECK-NEXT:  [ 8] .strtab           STRTAB          0000000000000000 0002db 000032 00      0   0  1

// CHECK:      Relocation section '.rela.dyn' at offset 0x158 contains 2 entries:
// CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
// CHECK-NEXT: 00000000010021f0  000000000000003d R_390_IRELATIVE                   1001188
// CHECK-NEXT: 00000000010021f8  000000000000003d R_390_IRELATIVE                   100118a

// CHECK:      Symbol table '.symtab' contains 6 entries:
// CHECK-NEXT:   Num:    Value          Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK-NEXT:     1: 0000000001000158     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
// CHECK-NEXT:     2: 0000000001000188     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end
// CHECK-NEXT:     3: 0000000001001188     0 IFUNC   GLOBAL DEFAULT     2 foo
// CHECK-NEXT:     4: 000000000100118a     0 IFUNC   GLOBAL DEFAULT     2 bar
// CHECK-NEXT:     5: 000000000100118c     0 NOTYPE  GLOBAL DEFAULT     2 _start

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <foo>:
// DISASM-NEXT:  1001188: br      %r14
// DISASM: <bar>:
// DISASM-NEXT:  100118a: br      %r14
// DISASM:      <_start>:
// DISASM-NEXT:  100118c: brasl   %r14, 0x10011b0
// DISASM-NEXT:  1001192: brasl   %r14, 0x10011d0
// DISASM-NEXT:  1001198: larl    %r2, 0x1000158
// DISASM-NEXT:  100119e: larl    %r2, 0x1000188
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: <.iplt>:
// DISASM:        10011b0:       larl    %r1, 0x10021f0
// DISASM-NEXT:   10011b6:       lg      %r1, 0(%r1)
// DISASM-NEXT:   10011bc:       br      %r1
// DISASM:        10011d0:       larl    %r1, 0x10021f8
// DISASM-NEXT:   10011d6:       lg      %r1, 0(%r1)
// DISASM-NEXT:   10011dc:       br      %r1

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 br %r14

.type bar STT_GNU_IFUNC
.globl bar
bar:
 br %r14

.globl _start
_start:
 brasl %r14, foo@plt
 brasl %r14, bar@plt
 larl %r2, __rela_iplt_start
 larl %r2, __rela_iplt_end
