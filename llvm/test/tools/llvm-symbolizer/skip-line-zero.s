##  Test  --skip-line-zero option.
##
##  This test illustrates the usage of handcrafted assembly to produce the following line table.
##  Address            Line   Column File   ISA Discriminator OpIndex Flags
##  ------------------ ------ ------ ------ --- ------------- ------- -------------
##  0x0000000000001710      1      0      2   0             0       0  is_stmt
##  0x0000000000001717      0     17      2   0             0       0  is_stmt prologue_end
##  0x000000000000171a      2     15      2   0             0       0
##  0x000000000000171f      2      3      2   0             0       0  epilogue_begin
##  0x0000000000001721      2      3      2   0             0       0  end_sequence
##  0x00000000000016c0      0      0      1   0             0       0  is_stmt
##  0x00000000000016cf      3     29      1   0             0       0  is_stmt prologue_end
##  0x00000000000016d5      0     25      1   0             0       0
##  0x00000000000016da      3     18      1   0             0       0  epilogue_begin
##  0x00000000000016e0      3     18      1   0             0       0  end_sequence

# REQUIRES: x86-registered-target

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux --fdebug-prefix-map=%t="" %s -o %t.o

## Check that without '--skip-line-zero', line number zero is displayed for the line-table entry which has no source correspondence.
# RUN: llvm-symbolizer --obj=%t.o 0x16d5 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-DISABLE %s

# APPROX-DISABLE:main
# APPROX-DISABLE-NEXT:main.c:0:25

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
## If it fails to find in the current sequence then return the original computed line-zero for the queried address.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x16c0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-FAIL-ACROSS-SEQ %s

# APPROX-FAIL-ACROSS-SEQ:main
# APPROX-FAIL-ACROSS-SEQ-NEXT:main.c:0:0

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-WITHIN-SEQ %s

# APPROX-WITHIN-SEQ:foo
# APPROX-WITHIN-SEQ-NEXT:definitions.c:1:0 (approximate)

## Check to ensure that '--skip-line-zero' only affects addresses having line-zero when more than one address is specified.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x16d5 0x16da | FileCheck --strict-whitespace --match-full-lines --check-prefixes=APPROX-ENABLE,NO-APPROX %s

# APPROX-ENABLE:main
# APPROX-ENABLE-NEXT:main.c:3:29 (approximate)
# NO-APPROX:main
# NO-APPROX-NEXT:main.c:3:18

## Check to ensure that '--skip-line-zero' with '--verbose' enabled displays correct approximate flag in verbose ouptut.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --verbose 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-VERBOSE %s

# APPROX-VERBOSE:foo
# APPROX-VERBOSE-NEXT:  Filename: definitions.c
# APPROX-VERBOSE-NEXT:  Function start filename: definitions.c
# APPROX-VERBOSE-NEXT:  Function start line: 1
# APPROX-VERBOSE-NEXT:  Function start address: 0x1710
# APPROX-VERBOSE-NEXT:  Line: 1
# APPROX-VERBOSE-NEXT:  Column: 0
# APPROX-VERBOSE-NEXT:  Approximate: true

## Check to ensure that '--skip-line-zero' with '--output-style=JSON' displays correct approximate flag in JSON output.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --output-style=JSON 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-JSON %s

# APPROX-JSON:[{"Address":"0x1717","ModuleName":"{{.*}}{{[/|\]+}}test{{[/|\]+}}tools{{[/|\]+}}llvm-symbolizer{{[/|\]+}}Output{{[/|\]+}}skip-line-zero.s.tmp.o","Symbol":[{"Approximate":true,"Column":0,"Discriminator":0,"FileName":"definitions.c","FunctionName":"foo","Line":1,"StartAddress":"0x1710","StartFileName":"definitions.c","StartLine":1}]}]

#--- definitions.c
#__attribute__((section("def"))) unsigned int foo(unsigned int x) {
#  return 1234 + x;
#}

#--- main.c
#include "definitions.c"
#unsigned int x = 1000;
#int main(void) { return foo(x); }

#--- gen
#clang -S -gdwarf-4 --target=x86_64-pc-linux -fdebug-prefix-map=/proc/self/cwd="" -fdebug-prefix-map=./="" main.c -o main.s

#sed -i '1,72d' main.s                              # Delete .text and .dummy_section
#sed -i '137c\	.quad 0x1710                         # DW_AT_low_pc' main.s
#sed -i '138c\	.long 0x1721-0x1710                  # DW_AT_high_pc' main.s
#sed -i '157c\	.quad 0x16c0                         # DW_AT_low_pc' main.s
#sed -i '158c\	.long 0x16e0-0x16c0                  # DW_AT_high_pc' main.s
#sed -i '175c\	.quad 0x1710' main.s
#sed -i '176c\	.quad 0x1721' main.s
#sed -i '177c\	.quad 0x16c0' main.s
#sed -i '178c\	.quad 0x16e0' main.s
#sed -i '$a\	.long .Lunit_end - .Lunit_start   # unit length\
#.Lunit_start:\
#	.short 4   # version\
#	.long .Lprologue_end - .Lprologue_start      # header length\
#.Lprologue_start:\
#	.byte 1                                      # minimum_instruction_length\
#	.byte 1                                      # maximum_operations_per_instruction\
#	.byte 1                                      # default_is_stmt\
#	.byte -5                                     # line_base\
#	.byte 14                                     # line_range\
#	.byte 13                                     # opcode_base\
#	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes\
#	.byte 0                                      # end of include directories\
#	.asciz "main.c"                              # filename\
#	.byte 0                                      # directory index\
#	.byte 0                                      # modification time\
#	.byte 0                                      # length of file (unavailable)\
#	.asciz "definitions.c"                       # filename\
#	.byte 0                                      # directory index\
#	.byte 0                                      # modification time\
#	.byte 0                                      # length of file (unavailable)\
#	.byte 0                                      # end of filenames\
#.Lprologue_end:\
#	.byte 0x04, 2                                # DW_LNS_set_file (2)\
#	.byte 0x00, 9, 2                             # DW_LNE_set_address\
#	.quad 0x1710                                 # Address Value\
#	.byte 0x01                                   # DW_LNS_copy\
#	.byte 0x05, 17                               # DW_LNS_set_column (17)\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0x73                                   # (address += 7,  line += -1,  op-index += 0)\
#	.byte 0x05, 15                               # DW_LNS_set_column (15)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x3e                                   # (address += 3,  line += 2,  op-index += 0)\
#	.byte 0x05, 3                                # DW_LNS_set_column\
#	.byte 0x0b                                   # DW_LNS_set_epilogue_begin\
#	.byte 0x58                                   # (address += 5,  line += 0,  op-index += 0)\
#	.byte 0x02                                   # DW_LNS_advance_pc\
#	.uleb128 0x02                                # (addr += 2, op-index += 0)\
#	.byte 0x00, 1, 1                             # DW_LNE_end_sequence\
#	.byte 0x00, 9, 2                             # DW_LNE_set_address\
#	.quad 0x16c0                                 # Address Value\
#	.byte 0x11                                   # (address += 0,  line += -1,  op-index += 0)\
#	.byte 0x05, 29                               # DW_LNS_set_column (29)\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0xe7                                   # (address += 15,  line += 3,  op-index += 0)\
#	.byte 0x05, 25                               # DW_LNS_set_column (25)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x63                                   # (address += 6,  line += -3,  op-index += 0)\
#	.byte 0x05, 18                               # DW_LNS_set_column (18)\
#	.byte 0x0b                                   # DW_LNS_set_epilogue_begin\
#	.byte 0x5b                                   # (address += 5,  line += 3,  op-index += 0)\
#	.byte 0x02                                   # DW_LNS_advance_pc\
#	.uleb128 0x06                                # (addr += 6, op-index += 0)\
#	.byte 0x00, 1, 1                             # DW_LNE_end_sequence\
#.Lunit_end:' main.s

#sed -n p main.s

#--- main.s
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x80 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x26:0x15 DW_TAG_variable
	.long	.Linfo_string2                  # DW_AT_name
	.long	59                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	x
	.byte	3                               # Abbrev [3] 0x3b:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x42:0x28 DW_TAG_subprogram
	.quad 0x1710                         # DW_AT_low_pc
	.long 0x1721-0x1710                  # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string4                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	59                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x5b:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string2                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	59                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x6a:0x19 DW_TAG_subprogram
	.quad 0x16c0                         # DW_AT_low_pc
	.long 0x16e0-0x16c0                  # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	131                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x83:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad 0x1710
	.quad 0x1721
	.quad 0x16c0
	.quad 0x16e0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"main.c"                        # string offset=1
.Linfo_string2:
	.asciz	"x"                             # string offset=8
.Linfo_string3:
	.asciz	"unsigned int"                  # string offset=10
.Linfo_string4:
	.asciz	"foo"                           # string offset=23
.Linfo_string5:
	.asciz	"main"                          # string offset=27
.Linfo_string6:
	.asciz	"int"                           # string offset=32
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym foo
	.addrsig_sym x
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	.long .Lunit_end - .Lunit_start   # unit length
.Lunit_start:
	.short 4   # version
	.long .Lprologue_end - .Lprologue_start      # header length
.Lprologue_start:
	.byte 1                                      # minimum_instruction_length
	.byte 1                                      # maximum_operations_per_instruction
	.byte 1                                      # default_is_stmt
	.byte -5                                     # line_base
	.byte 14                                     # line_range
	.byte 13                                     # opcode_base
	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes
	.byte 0                                      # end of include directories
	.asciz "main.c"                              # filename
	.byte 0                                      # directory index
	.byte 0                                      # modification time
	.byte 0                                      # length of file (unavailable)
	.asciz "definitions.c"                       # filename
	.byte 0                                      # directory index
	.byte 0                                      # modification time
	.byte 0                                      # length of file (unavailable)
	.byte 0                                      # end of filenames
.Lprologue_end:
	.byte 0x04, 2                                # DW_LNS_set_file (2)
	.byte 0x00, 9, 2                             # DW_LNE_set_address
	.quad 0x1710                                 # Address Value
	.byte 0x01                                   # DW_LNS_copy
	.byte 0x05, 17                               # DW_LNS_set_column (17)
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0x73                                   # (address += 7,  line += -1,  op-index += 0)
	.byte 0x05, 15                               # DW_LNS_set_column (15)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x3e                                   # (address += 3,  line += 2,  op-index += 0)
	.byte 0x05, 3                                # DW_LNS_set_column
	.byte 0x0b                                   # DW_LNS_set_epilogue_begin
	.byte 0x58                                   # (address += 5,  line += 0,  op-index += 0)
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x02                                # (addr += 2, op-index += 0)
	.byte 0x00, 1, 1                             # DW_LNE_end_sequence
	.byte 0x00, 9, 2                             # DW_LNE_set_address
	.quad 0x16c0                                 # Address Value
	.byte 0x11                                   # (address += 0,  line += -1,  op-index += 0)
	.byte 0x05, 29                               # DW_LNS_set_column (29)
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0xe7                                   # (address += 15,  line += 3,  op-index += 0)
	.byte 0x05, 25                               # DW_LNS_set_column (25)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x63                                   # (address += 6,  line += -3,  op-index += 0)
	.byte 0x05, 18                               # DW_LNS_set_column (18)
	.byte 0x0b                                   # DW_LNS_set_epilogue_begin
	.byte 0x5b                                   # (address += 5,  line += 3,  op-index += 0)
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x06                                # (addr += 6, op-index += 0)
	.byte 0x00, 1, 1                             # DW_LNE_end_sequence
.Lunit_end:
