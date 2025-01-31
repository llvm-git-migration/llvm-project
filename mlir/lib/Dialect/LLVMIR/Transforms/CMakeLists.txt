set(LLVM_OPTIONAL_SOURCES
  AddComdats.cpp
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp
)

add_mlir_dialect_library(MLIRLLVMIRTransformsLegalizeForExport
  LegalizeForExport.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)

add_mlir_dialect_library(MLIRLLVMIRTransforms
  AddComdats.cpp
  DIExpressionLegalization.cpp
  DIExpressionRewriter.cpp
  DIScopeForLLVMFuncOp.cpp
  InlinerInterfaceImpl.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransformsLegalizeForExport
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect
  )
