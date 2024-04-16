set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(offload-tblgen Offload
  EXPORT Offload
  APIGen.cpp
  offload-tblgen.cpp
  ValidationGen.cpp
  )
