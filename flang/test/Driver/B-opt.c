// Check -B driver option.

/// Target triple prefix is not detected for -B.
// RUN: %flang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir1 -fuse-ld=ld 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-TRIPLE %s
// CHECK-B-OPT-TRIPLE-NOT: "{{.*}}/Inputs/B_opt_tree/dir1{{/|\\\\}}i386-unknown-linux-ld"
//
// RUN: %flang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/B_opt_tree/dir2 -fuse-ld=ld 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-B-OPT-DIR %s
// CHECK-B-OPT-DIR: "{{.*}}/Inputs/B_opt_tree/dir2{{/|\\\\}}ld" 
