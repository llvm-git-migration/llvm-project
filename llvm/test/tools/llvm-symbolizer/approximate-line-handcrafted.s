# REQUIRES: x86-registered-target

# RUN: clang -O3 -gline-tables-only --target=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o 0x1730 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-FAIL-WITHIN-SEQ %s
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x17a0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-WITHIN-SEQ %s
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x17a0 0x1757 | FileCheck --strict-whitespace --match-full-lines --check-prefixes=APPROX-WITHIN-SEQ,NO-APPROX %s
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --verbose 0x17a0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-VERBOSE %s
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --output-style=JSON 0x17a0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-JSON %s

# APPROX-FAIL-WITHIN-SEQ:main
# APPROX-FAIL-WITHIN-SEQ-NEXT:{{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:0:0
# APPROX-WITHIN-SEQ:sub
# APPROX-WITHIN-SEQ-NEXT:{{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}.{{[/|\]+}}definitions.h:1:80 (approximate)
# NO-APPROX:main
# NO-APPROX-NEXT:{{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}main.c:9:3

#APPROX-VERBOSE:sub
#APPROX-VERBOSE-NEXT:  Filename: /tmp/test/.{{[/|\]}}definitions.h
#APPROX-VERBOSE-NEXT:  Function start address: 0x17a0
#APPROX-VERBOSE-NEXT:  Line: 1
#APPROX-VERBOSE-NEXT:  Column: 80
#APPROX-VERBOSE-NEXT:  Approximate: 1

#APPROX-JSON:[{"Address":"0x17a0","ModuleName":"{{.*}}{{[/|\]+}}test{{[/|\]+}}tools{{[/|\]+}}llvm-symbolizer{{[/|\]+}}Output{{[/|\]+}}approximate-line-handcrafted.s.tmp.o","Symbol":[{"Approximate":true,"Column":80,"Discriminator":0,"FileName":"{{[/|\]+}}tmp{{[/|\]+}}test{{[/|\]+}}.{{[/|\]+}}definitions.h","FunctionName":"sub","Line":1,"StartAddress":"0x17a0","StartFileName":"","StartLine":0}]}]

## Generated from C Code
##
## // definitions.h
## extern inline __attribute__((section(".def_section"))) int add(int x, int y) { return (x + y); }
## extern inline __attribute__((section(".def_section"))) int sub(int x, int y) { return (x - y); }
##
## // main.c
## include <stdio.h>
## include "definitions.h"
##
## int main(void) {
## int a = 10;
## int b = 100;
## printf("Addition result: %d \n", add(a, b));
## printf("Subtraction result: %d \n", sub(a, b));
## return 0;
## }
##
## clang -S -O3 -gline-tables-only --target=x86_64-pc-linux
##
## The assembly generated here is modified manually.
## Manual Edited Entry-1 : Line 78 changed to ".loc   1 0 0 is_stmt 1". Original ".loc   1 2 0 is_stmt 1"
## Manual Edited Entry-2 : Line 98 changed to ".loc   0 0 0 is_stmt 1". Original ".loc   0 4 0 is_stmt 1"

	.text
	.file	"main.c"
	.section	.def_section,"ax",@progbits
	.globl	add                             # -- Begin function add
	.p2align	4, 0x90
	.type	add,@function
add:                                    # @add
.Lfunc_begin0:
	.file	0 "/tmp/test" "main.c" md5 0xd17012b9cce0f9e2899a6021bde62f2b
	.cfi_startproc
# %bb.0:                                # %entry
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	.file	1 "." "definitions.h" md5 0xad3977bad33342557e615dded17b720b
	.loc	1 1 90 prologue_end             # ./definitions.h:1:90
	leal	(%rdi,%rsi), %eax
	.loc	1 1 80 is_stmt 0                # ./definitions.h:1:80
	retq
.Ltmp0:
.Lfunc_end0:
	.size	add, .Lfunc_end0-add
	.cfi_endproc
                                        # -- End function
	.globl	sub                             # -- Begin function sub
	.p2align	4, 0x90
	.type	sub,@function
sub:                                    # @sub
.Lfunc_begin1:
	.loc	1 0 0 is_stmt 1                 # ./definitions.h:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edi, %eax
.Ltmp1:
	.loc	1 2 90 prologue_end             # ./definitions.h:2:90
	subl	%esi, %eax
	.loc	1 2 80 is_stmt 0                # ./definitions.h:2:80
	retq
.Ltmp2:
.Lfunc_end1:
	.size	sub, .Lfunc_end1-sub
	.cfi_endproc
                                        # -- End function
	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin2:
	.loc	0 0 0 is_stmt 1                 # main.c:4:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
.Ltmp3:
	.loc	0 7 3 prologue_end              # main.c:7:3
	leaq	.L.str(%rip), %rdi
	movl	$110, %esi
	xorl	%eax, %eax
	callq	printf@PLT
.Ltmp4:
	.loc	0 8 3                           # main.c:8:3
	leaq	.L.str.1(%rip), %rdi
	movl	$-90, %esi
	xorl	%eax, %eax
	callq	printf@PLT
.Ltmp5:
	.loc	0 9 3                           # main.c:9:3
	xorl	%eax, %eax
	.loc	0 9 3 epilogue_begin is_stmt 0  # main.c:9:3
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Ltmp6:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Addition result: %d \n"
	.size	.L.str, 22

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"Subtraction result: %d \n"
	.size	.L.str.1, 25

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x1f DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	29                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin0      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .Lfunc_end2-.Lfunc_begin2      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	16                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git b93be88806ad3851d4f0dca37c7c5ca7f3e92654)" # string offset=0
.Linfo_string1:
	.asciz	"main.c"                        # string offset=113
.Linfo_string2:
	.asciz	"/tmp/test"                     # string offset=120
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin2
.Ldebug_addr_end0:
	.ident	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git b93be88806ad3851d4f0dca37c7c5ca7f3e92654)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
