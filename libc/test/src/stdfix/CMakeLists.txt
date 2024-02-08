if(NOT LIBC_COMPILER_HAS_FIXED_POINT)
  return()
endif()

foreach(suffix IN hr r lr hk k lk)
  add_libc_test(
    abs${suffix}_test
    HDRS
      AbsTest.h
    SRCS
      abs${suffix}_test.cpp
    COMPILE_OPTIONS
      -O3
    DEPENDS
      libc.src.stdfix.abs${suffix}
      libc.src.__support.fixedpoint.fx_bits
  )
endforeach()
