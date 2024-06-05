#include "llvm/ADT/MPInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

hash_code llvm::hash_value(const MPInt &X) {
  if (X.isSmall())
    return llvm::hash_value(X.getSmall());
  return detail::hash_value(X.getLarge());
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
raw_ostream &MPInt::print(raw_ostream &OS) const {
  if (isSmall())
    return OS << ValSmall;
  return OS << ValLarge;
}

void MPInt::dump() const { print(dbgs()); }
#endif
