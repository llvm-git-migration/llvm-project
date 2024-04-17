# Check if OMPT support is available
# Currently, __builtin_frame_address() is required for OMPT
# Weak attribute is required for Unices (except Darwin), LIBPSAPI is used for Windows
if(NOT LIBOMP_ARCH)
  include(LibompGetArchitecture)
  libomp_get_architecture(LIBOMP_DETECTED_ARCH)
  set(LIBOMP_ARCH ${LIBOMP_DETECTED_ARCH})
endif()

check_c_source_compiles("int main(int argc, char** argv) {
  void* p = __builtin_frame_address(0);
  return 0;}" LIBOMP_HAVE___BUILTIN_FRAME_ADDRESS)
check_c_source_compiles("__attribute__ ((weak)) int foo(int a) { return a*a; }
  int main(int argc, char** argv) {
  return foo(argc);}" LIBOMP_HAVE_WEAK_ATTRIBUTE)
set(CMAKE_REQUIRED_LIBRARIES psapi)
check_c_source_compiles("#include <windows.h>
  #include <psapi.h>
  int main(int artc, char** argv) {
    return EnumProcessModules(NULL, NULL, 0, NULL);
  }" LIBOMP_HAVE_PSAPI)
set(CMAKE_REQUIRED_LIBRARIES)
if(NOT LIBOMP_HAVE___BUILTIN_FRAME_ADDRESS)
  set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
else()
  if( # hardware architecture supported?
     ((LIBOMP_ARCH STREQUAL x86_64) OR
      (LIBOMP_ARCH STREQUAL i386) OR
#      (LIBOMP_ARCH STREQUAL arm) OR
      (LIBOMP_ARCH STREQUAL aarch64) OR
      (LIBOMP_ARCH STREQUAL aarch64_32) OR
      (LIBOMP_ARCH STREQUAL aarch64_a64fx) OR
      (LIBOMP_ARCH STREQUAL ppc64le) OR
      (LIBOMP_ARCH STREQUAL ppc64) OR
      (LIBOMP_ARCH STREQUAL riscv64) OR
      (LIBOMP_ARCH STREQUAL loongarch64) OR
      (LIBOMP_ARCH STREQUAL s390x))
     AND # OS supported?
     ((WIN32 AND LIBOMP_HAVE_PSAPI) OR APPLE OR
      (NOT (WIN32 OR ${CMAKE_SYSTEM_NAME} MATCHES "AIX") AND LIBOMP_HAVE_WEAK_ATTRIBUTE)))
    set(LIBOMP_HAVE_OMPT_SUPPORT TRUE)
  else()
    set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
  endif()
endif()

# OMPT-support defaults to ON for OpenMP 5.0+ and if the requirements in
# cmake/config-ix.cmake are fulfilled.
set(OMPT_DEFAULT FALSE)
if ((LIBOMP_HAVE_OMPT_SUPPORT) AND (NOT WIN32))
  set(OMPT_DEFAULT TRUE)
endif()
set(LIBOMP_OMPT_SUPPORT ${OMPT_DEFAULT} CACHE BOOL
  "OMPT-support?")

# Check LIBOMP_HAVE_VERSION_SCRIPT_FLAG
include(LLVMCheckCompilerLinkerFlag)
if(NOT APPLE)
  llvm_check_compiler_linker_flag(C "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/../../openmp/runtime/src/exports_test_so.txt" LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
endif()

macro(pythonize_bool var)
if (${var})
  set(${var} True)
else()
  set(${var} False)
endif()
endmacro()
