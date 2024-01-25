add_mlir_dialect(MeshOps mesh)
add_mlir_doc(MeshOps MeshOps Dialects/ -gen-dialect-doc -dialect=mesh)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MeshAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMeshAttrIncGen)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshEnums.h.inc -gen-enum-decls)
mlir_tablegen(MeshEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMeshEnumsIncGen)
