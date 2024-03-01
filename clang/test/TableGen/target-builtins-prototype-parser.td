// RUN: clang-tblgen -I %p/../../../clang/include/ %s --gen-clang-builtins | FileCheck %s
// RUN: not clang-tblgen -I %p/../../../clang/include/ %s --gen-clang-builtins -DERROR_EXPECTED_LANES 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_LANES

include "clang/Basic/BuiltinsBase.td"

def : Builtin {
// CHECK: BUILTIN(__builtin_test_use_clang_extended_vectors, "E8idE4b", "")
  let Prototype = "ext_vector_type_8_int(double, ext_vector_type_4_bool)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}


#ifdef ERROR_EXPECTED_LANES
def : Builtin {
// ERROR_EXPECTED_LANES: :[[# @LINE + 1]]:7: error: Expected number of lanes
  let Prototype = "ext_vector_type__int(double, ext_vector_type_4_bool)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

