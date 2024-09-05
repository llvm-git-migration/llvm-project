add_llvm_component_library(LLVMSandboxVec
  SandboxVectorizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  SandboxIR
  Support
  TransformUtils
)
