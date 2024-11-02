# llvm-exegesis

`llvm-exegesis` is a benchmarking tool that accepts or generates snippets and
can measure characteristics of those snippets by executing it while keeping track
of performance counters.

### Currently Supported Platforms

`llvm-exegesis` is quite platform-dependent and currently only supports a couple
platform configurations for benchmarking. The limitations are listed below.
Analysis mode in `llvm-exegesis` is supported on all platforms on which LLVM is.

#### Currently Supported Operating Systems for Benchmarking

Currently, `llvm-exegesis`  only supports benchmarking on Linux. This is mainly
due to a dependency on the Linux perf subsystem and libpfm4 for reading
performance counters. For benchmarking, LLVM needs to be configured and build
with `LLVM_ENABLE_LIBPFM` enabled. Benchmarking might fail if the target cpu is
not supported by libpfm. This can be checked by putting libpfm in verbose/debug
mode with environment variables `LIBPFM_VERBOSE` and `LIBFM_DEBUG`. If libpfm
is installed in a non-standard location, LLVM can be configured to look for the
libpfm library and header file locations by setting environment variables
`LIBRARY_PATH`, `C_INCLUDE_PATH`, and `CPLUS_INCLUDE_PATH`. In that case variable
`LD_LIBRARY_PATH` needs to be set too so that `llvm-exegesis` can find the
library at execution time.

The subprocess execution mode and memory annotations currently only supports
Linux due to a heavy reliance on many Linux specific syscalls/syscall
implementations.

#### Currently Supported Architectures for Benchmarking

Currently, using `llvm-exegesis` for benchmarking is supported on the following
architectures:
* x86
  * 64-bit only due to this being the only implemented calling convention
    in `llvm-exegesis` currently.
* ARM
  * Very experimental AArch64 support only, most opcodes probably won't work.
* MIPS
* PowerPC (PowerPC64LE only)

Note that not all benchmarking functionality is guaranteed to work on all platforms.

Memory annotations are currently only supported on 64-bit X86. There is no
inherent limitations for porting memory annotations to other architectures, but
parts of the test harness are implemented as MCJITed assembly that is generated
in `./lib/X86/Target.cpp` that would need to be implemented on other architectures
to bring up support.
