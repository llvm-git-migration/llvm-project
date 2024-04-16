
# TODO: #75125 isn't merged yet, use some hacky CMake to get things building in
# the mean time.
if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

include(${LLVM_COMMON_CMAKE_UTILS}/../llvm/cmake/modules/TableGen.cmake)
###


add_subdirectory(tools/offload-tblgen)

