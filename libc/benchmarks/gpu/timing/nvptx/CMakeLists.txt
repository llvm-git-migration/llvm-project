message(STATUS "IN nvptx dir")
add_header_library(
  nvptx_timing
  HDRS
    timing.h
  DEPENDS
    libc.src.__support.common
)
