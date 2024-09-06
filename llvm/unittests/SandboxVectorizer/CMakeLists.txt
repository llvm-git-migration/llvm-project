set(LLVM_LINK_COMPONENTS
  AsmParser
  SandboxIR
  Core
  )

add_llvm_unittest(SandboxVectorizerTests
  PassTest.cpp
  )
