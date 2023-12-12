/// Check driver handling for "-fvisibility-global-new-delete-hidden" and "-fvisibility-global-new-delete".

/// These options are not added by default.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DEFAULTS %s
// DEFAULTS-NOT: "-fvisibility-global-new-delete"
// DEFAULTS-NOT: "-fno-visibility-global-new-delete"
// DEFAULTS-NOT: "-fvisibility-global-new-delete-hidden"

// DEFINE: %{implicit-check-nots} = --implicit-check-not=-fvisibility-global-new-delete --implicit-check-not=-fno-visibility-global-new-delete

/// "-fno-visibility-global-new-delete" added by default for PS5.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=PS5 %s 
// PS5: "-fno-visibility-global-new-delete"

/// -fvisibility-global-new-delete-hidden added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=HIDDEN %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=HIDDEN %s %{implicit-check-nots}
// HIDDEN-DAG: "-fvisibility-global-new-delete-hidden"

/// -fvisibility-global-new-delete added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete %s 2>&1 | FileCheck -check-prefixes=VGND %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete %s 2>&1 | FileCheck -check-prefixes=VGND %s %{implicit-check-nots}
// VGND-DAG: "-fvisibility-global-new-delete"

/// -fno-visibility-global-new-delete added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fno-visibility-global-new-delete %s 2>&1 | FileCheck -check-prefixes=NO_VGND %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fno-visibility-global-new-delete %s 2>&1 | FileCheck -check-prefixes=NO_VGND %s %{implicit-check-nots}
// NO_VGND-DAG: "-fno-visibility-global-new-delete"

/// No error if both -fvisibility-global-new-delete and -fvisibility-global-new-delete-hidden specified.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete-hidden -fvisibility-global-new-delete %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=VGND,HIDDEN %s %{implicit-check-nots}

/// Error if both -fno-visibility-global-new-delete and -fvisibility-global-new-delete-hidden specified.
// RUN: not %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete-hidden -fvisibility-global-new-delete -fno-visibility-global-new-delete %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=INCOMPAT %s
// INCOMPAT: clang: error: the combination of '-fno-visibility-global-new-delete' and '-fvisibility-global-new-delete-hidden' is incompatible
