macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

function(get_all_targets dir outvar)
    set(targets)
    get_all_targets_recursive(targets ${dir})
    set(${outvar} ${targets} PARENT_SCOPE)
endfunction()

function(add_llvm_lib_precompiled_headers target)
    if (LLVM_ENABLE_PRECOMPILED_HEADERS)
        get_target_property(target_type ${target} TYPE)
        if (target_type STREQUAL "STATIC_LIBRARY")
            target_precompile_headers(
                ${target}
                PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:${LLVM_MAIN_INCLUDE_DIR}/llvm/PrecompiledHeaders.h>"
            )
        endif()
    endif()
endfunction()

function(llvm_lib_precompiled_headers)
    get_all_targets("${LLVM_MAIN_SRC_DIR}/lib" lib_targets)
    foreach(target ${lib_targets})
        add_llvm_lib_precompiled_headers(${target})
    endforeach()
endfunction()
