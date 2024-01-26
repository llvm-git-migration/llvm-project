set(full_build_flag "")
if(LLVM_LIBC_FULL_BUILD)
  set(full_build_flag "-DLIBC_FULL_BUILD")
endif()

add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  HDRS
    libc_errno.h     # Include this
  COMPILE_OPTIONS
    ${full_build_flag}
  DEPENDS
    libc.include.errno
    libc.src.__support.common
)
