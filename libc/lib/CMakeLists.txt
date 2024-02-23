set(libc_archive_targets "")
set(libc_archive_names "")
set(libc_archive_entrypoint_lists "")
if(LLVM_LIBC_FULL_BUILD)
  list(APPEND libc_archive_names c m)
  list(APPEND libc_archive_targets libc libm)
  list(APPEND libc_archive_entrypoint_lists
       TARGET_LIBC_ENTRYPOINTS TARGET_LIBM_ENTRYPOINTS)
else()
  list(APPEND libc_archive_names llvmlibc)
  list(APPEND libc_archive_targets libc)
  list(APPEND libc_archive_entrypoint_lists TARGET_LLVMLIBC_ENTRYPOINTS)
endif()

set(added_archive_targets "")
foreach(archive IN ZIP_LISTS
        libc_archive_names libc_archive_targets libc_archive_entrypoint_lists)
  if(NOT ${archive_2})
    # If an entrypoint list is missing, then skip adding targets for it.
    continue()
  endif()
  add_entrypoint_library(
    ${archive_1}
    DEPENDS
      ${${archive_2}}
  )
  set_target_properties(
    ${archive_1}
    PROPERTIES
      ARCHIVE_OUTPUT_NAME ${archive_0}
  )
  if(LLVM_LIBC_FULL_BUILD)
    target_link_libraries(${archive_1} PUBLIC libc-headers)
    if(TARGET libc-startup)
      add_dependencies(${archive_1} libc-startup)
    endif()
  endif()
  list(APPEND added_archive_targets ${archive_1})

  # The GPU build additionally exports the libraries as both a fat binary
  # archive and an LLVM-IR bitcode blob.
  # FIXME: These don't properly re-run if the source gets modified. We willy
  #        likely need source file dependencies for them.
  if(LIBC_TARGET_OS_IS_GPU)
    add_gpu_entrypoint_library(
      ${archive_1}.__fatbin__
      DEPENDS
        ${${archive_2}}
    )
    set_target_properties(
      ${archive_1}.__fatbin__
      PROPERTIES
        ARCHIVE_OUTPUT_NAME ${archive_0}gpu-${LIBC_TARGET_ARCHITECTURE}
        ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
    )

    add_bitcode_entrypoint_library(
      ${archive_1}.__bitcode__
      DEPENDS
        ${${archive_2}}
    )
    add_dependencies(${archive_1} ${archive_1}.__bitcode__)

    install(
      TARGETS ${added_gpu_archive_targets}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      COMPONENT libc
    )

    install(FILES $<TARGET_PROPERTY:${archive_1}.__bitcode__,TARGET_OBJECT>
            DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
            RENAME ${archive_1}.bc
            COMPONENT libc
    )
  endif()
endforeach()

install(
  TARGETS ${added_archive_targets}
  ARCHIVE DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT libc
)

if(LIBC_TARGET_OS_IS_GPU)
endif()

if(NOT LIBC_TARGET_OS_IS_BAREMETAL)
  # For now we will disable libc-startup installation for baremetal. The
  # correct way to do it would be to make a hookable startup for baremetal
  # and install it as part of the libc installation.
  set(startup_target "libc-startup")
endif()

if(LLVM_LIBC_FULL_BUILD)
  set(header_install_target install-libc-headers)
endif()

add_custom_target(install-libc
                  DEPENDS ${added_archive_targets}
                          ${startup_target}
                          ${header_install_target}
                  COMMAND "${CMAKE_COMMAND}"
                          -DCMAKE_INSTALL_COMPONENT=libc
                          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
add_custom_target(install-libc-stripped
                  DEPENDS ${added_archive_targets}
                          ${startup_target}
                          ${header_install_target}
                  COMMAND "${CMAKE_COMMAND}"
                          -DCMAKE_INSTALL_COMPONENT=libc
                          -DCMAKE_INSTALL_DO_STRIP=1
                          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
