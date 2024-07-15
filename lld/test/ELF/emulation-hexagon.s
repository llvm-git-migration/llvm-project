# REQUIRES: hexagon

# RUN: llvm-mc -filetype=obj -triple=hexagon %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefixes=HEXAGON,LE %s
# RUN: ld.lld -m elf32_littlehexagon %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefixes=HEXAGON,LE %s
# RUN: echo 'OUTPUT_FORMAT(elf32-hexagon)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefixes=HEXAGON,LE %s

# HEXAGON:       ElfHeader {
# HEXAGON-NEXT:    Ident {
# HEXAGON-NEXT:      Magic: (7F 45 4C 46)
# HEXAGON-NEXT:      Class: 32-bit (0x1)
# LE-NEXT:           DataEncoding: LittleEndian (0x1)
# BE-NEXT:           DataEncoding: BigEndian (0x2)
# HEXAGON-NEXT:      FileVersion: 1
# HEXAGON-NEXT:      OS/ABI: SystemV (0x0)
# HEXAGON-NEXT:      ABIVersion: 0
# HEXAGON-NEXT:      Unused: (00 00 00 00 00 00 00)
# HEXAGON-NEXT:    }
# HEXAGON-NEXT:    Type: Executable (0x2)
# HEXAGON-NEXT:    Machine: EM_HEXAGON (0xA4)
# HEXAGON-NEXT:    Version: 1
# HEXAGON-NEXT:    Entry: 0x200B4
# HEXAGON-NEXT:    ProgramHeaderOffset: 0x34
# HEXAGON-NEXT:    SectionHeaderOffset:
# HEXAGON-NEXT:    Flags [ (0x60)
# HEXAGON-NEXT:      0x20
# HEXAGON-NEXT:      0x40
# HEXAGON-NEXT:    ]

.globl _start
_start:
