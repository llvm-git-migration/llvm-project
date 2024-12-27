#include "src/stdlib/free.h"
#include "src/__support/alloc/alloc.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, free, (void *ptr)) {
  LIBC_ASSERT(allocator->free(ptr));
}

} // namespace LIBC_NAMESPACE_DECL
