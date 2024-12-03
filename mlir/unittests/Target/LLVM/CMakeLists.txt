if (LLVM_LINK_LLVM_DYLIB)
  llvm_map_components_to_libnames(LLVM_LINK_COMPONENTS nativecodegen)
else()
  llvm_map_components_to_libnames(llvm_libs nativecodegen)
endif()

add_mlir_unittest(MLIRTargetLLVMTests
  SerializeNVVMTarget.cpp
  SerializeROCDLTarget.cpp
  SerializeToLLVMBitcode.cpp
)

target_link_libraries(MLIRTargetLLVMTests
  PRIVATE
  MLIRTargetLLVM
  MLIRNVVMTarget
  MLIRROCDLTarget
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  ${llvm_libs}
)

if (DEFINED LLVM_NATIVE_TARGET)
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=1
  )
else()
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=0
  )
endif()
