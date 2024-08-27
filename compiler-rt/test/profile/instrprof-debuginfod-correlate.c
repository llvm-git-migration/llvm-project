// REQUIRES: linux || windows
// RUN: rm -rf %t

// Default build with no profile correlation.
// RUN: %clang_profgen -o %t.default.exe -Wl,--build-id=0x12345678 -fprofile-instr-generate -fcoverage-mapping %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.default.profraw %run %t.default.exe
// RUN: llvm-profdata merge -o %t.default.profdata %t.default.profraw

// Build with profile binary correlation and test llvm-profdata merge profile correlation with --binary-file option.
// RUN: %clang_profgen -o %t.correlate.exe -Wl,--build-id=0x12345678 -fprofile-instr-generate -fcoverage-mapping -mllvm -profile-correlate=binary %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// Strip above binary and run
// RUN: llvm-strip %t.correlate.exe -o %t.stripped.exe
// RUN: env LLVM_PROFILE_FILE=%t.correlate.profraw %run %t.stripped.exe
// RUN: llvm-profdata merge -o %t.correlate-binary.profdata --binary-file=%t.correlate.exe %t.correlate.profraw
// RUN: diff %t.default.profdata %t.correlate-binary.profdata

// Test llvm-profdata merge profile correlation with --debuginfod option.
// RUN: mkdir -p %t/buildid/12345678
// RUN: cp %t.correlate.exe %t/buildid/12345678/debuginfo
// RUN: env DEBUGINFOD_CACHE_PATH=%t/debuginfod-cache DEBUGINFOD_URLS=file://%t llvm-profdata merge -o %t.correlate-debuginfod.profdata --debuginfod %t.correlate.profraw
// RUN: diff %t.default.profdata %t.correlate-debuginfod.profdata

// Test llvm-profdata merge profile correlation with --debug-file-directory option.
// RUN: mkdir -p %t/.build-id/12
// RUN: cp %t.correlate.exe %t/.build-id/12/345678.debug
// RUN: llvm-profdata merge -o %t.correlate-debug-file-dir.profdata --debug-file-directory %t %t.correlate.profraw
// RUN: diff %t.default.profdata %t.correlate-debug-file-dir.profdata
