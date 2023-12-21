/// Check driver handling for "-fvisibility-global-new-delete-hidden" and "-f[no-]forced-global-new-delete-visibility".

/// These options are not added by default.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DEFAULTS %s
// DEFAULTS-NOT: "-fforced-global-new-delete-visibility"
// DEFAULTS-NOT: "-fno-forced-global-new-delete-visibility"
// DEFAULTS-NOT: "-fvisibility-global-new-delete-hidden"

// DEFINE: %{implicit-check-nots} = --implicit-check-not=-fforced-global-new-delete-visibility --implicit-check-not=-fno-forced-global-new-delete-visibility --implicit-check-not=-fvisibility-global-new-delete-hidden

/// "-fno-forced-global-new-delete-visibility" added by default for PS5.
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=PS5 %s 
// PS5: "-fno-forced-global-new-delete-visibility"

/// -fvisibility-global-new-delete-hidden added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=HIDDEN %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=HIDDEN %s %{implicit-check-nots}
// HIDDEN-DAG: "-fvisibility-global-new-delete-hidden"

/// -fforced-global-new-delete-visibility added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fforced-global-new-delete-visibility %s 2>&1 | FileCheck -check-prefixes=FGNDV %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fforced-global-new-delete-visibility %s 2>&1 | FileCheck -check-prefixes=FGNDV %s %{implicit-check-nots}
// FGNDV-DAG: "-fforced-global-new-delete-visibility"

/// -fno-forced-global-new-delete-visibility added explicitly.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \
// RUN:   -fno-forced-global-new-delete-visibility %s 2>&1 | FileCheck -check-prefixes=NO_FGNDV %s %{implicit-check-nots}
// RUN: %clang -### -target x86_64-sie-ps5 -x cl -c -emit-llvm \
// RUN:   -fno-forced-global-new-delete-visibility %s 2>&1 | FileCheck -check-prefixes=NO_FGNDV %s %{implicit-check-nots}
// NO_FGNDV-DAG: "-fno-forced-global-new-delete-visibility"

/// No error if both -fforced-global-new-delete-visibility and -fvisibility-global-new-delete-hidden specified.
// RUN: %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete-hidden -fforced-global-new-delete-visibility %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=FGNDV,HIDDEN %s %{implicit-check-nots}

/// Error if both -fno-forced-global-new-delete-visibility and -fvisibility-global-new-delete-hidden specified.
// RUN: not %clang -### -target x86_64-unknown-unknown -x cl -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete-hidden -fforced-global-new-delete-visibility -fno-forced-global-new-delete-visibility %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=INCOMPAT %s
// INCOMPAT: clang: error: the combination of '-fno-forced-global-new-delete-visibility' and '-fvisibility-global-new-delete-hidden' is incompatible
