// RUN: llvm-mc -filetype=obj -triple riscv32-unknown-linux-gnu -mattr=+experimental-zicfilp %s -o - | llvm-readelf --notes - | FileCheck %s --check-prefix=GNU
// RUN: llvm-mc -filetype=obj -triple riscv32-unknown-linux-gnu -mattr=+experimental-zicfilp %s -o - | llvm-readobj --notes - | FileCheck %s --check-prefix=LLVM

// GNU: Displaying notes found in: .note.gnu.property
// GNU-NEXT:   Owner                 Data size	Description
// GNU-NEXT:   GNU                   0x{{([0-9a-z]{8})}}	NT_GNU_PROPERTY_TYPE_0 (property note)
// GNU-NEXT:     Properties:    riscv feature: ZICFILP-func-sig

// LLVM:      NoteSections [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.gnu.property
// LLVM-NEXT:     Offset:
// LLVM-NEXT:     Size:
// LLVM-NEXT:     Notes [
// LLVM-NEXT:       {
// LLVM-NEXT:         Owner: GNU
// LLVM-NEXT:         Data size:
// LLVM-NEXT:         Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// LLVM-NEXT:         Property [
// LLVM-NEXT:           riscv feature: ZICFILP-func-sig
// LLVM-NEXT:         ]
// LLVM-NEXT:       }
// LLVM-NEXT:	    ]
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.gnu.property", "a"
  .long 4           /* n_namsz: always 4 (sizeof("GNU")) */
  .long end - begin /* n_descsz */
  .long 5           /* n_type: NT_GNU_PROPERTY_TYPE_0 */
  .asciz "GNU"      /* n_name */
begin:
  .long 0xc0000000  /* pr_type: GNU_PROPERTY_RISCV_FEATURE_1_AND */
  .long 4           /* pr_datasz */
  .long 4           /* pr_data: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG */
  .p2align 2        /* pr_padding */
end:
