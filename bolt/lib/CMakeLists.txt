add_compile_definitions(CMAKE_INSTALL_LIBDIR="${CMAKE_INSTALL_LIBDIR}"
  BOLT_RT_BINARY_DIR="${BOLT_BINARY_DIR}/bolt_rt-bins")

add_subdirectory(Core)
add_subdirectory(Passes)
add_subdirectory(Profile)
add_subdirectory(Rewrite)
add_subdirectory(RuntimeLibs)
add_subdirectory(Target)
add_subdirectory(Utils)
