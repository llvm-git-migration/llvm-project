#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>

using namespace llvm;

namespace {

struct TestCase {
  int64_t FixedImm;
  int64_t ScalableImm;
  bool Result;
};

const std::initializer_list<TestCase> Tests = {
    // FixedImm, ScalableImm, Result
    // No change, easily 'supported'
    {0, 0, true},

    // Simple fixed immediate cases
    // +8
    {8, 0, true},
    // -16
    {-16, 0, true},

    // Scalable; addvl increments by whole registers, range [-32,31]
    // +(16 * vscale), one register's worth
    {0, 16, true},
    // +(8 * vscale), half a register's worth
    {0, 8, false},
    // -(32 * 16 * vscale)
    {0, -512, true},
    // -(33 * 16 * vscale)
    {0, -528, false},
    // +(31 * 16 * vscale)
    {0, 496, true},
    // +(32 * 16 * vscale)
    {0, 512, false},

    // Mixed; not supported.
    // +(16 + (16 * vscale)) -- one register's worth + 16
    {16, 16, false},
};
} // namespace

TEST(Immediates, Immediates) {
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  auto TT = Triple::normalize("aarch64");
  const Target *T = TargetRegistry::lookupTarget(TT, Error);

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(TT, "generic", "+sve", TargetOptions(),
                             std::nullopt, std::nullopt,
                             CodeGenOptLevel::Default));
  AArch64Subtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                      TM->getTargetCPU(), TM->getTargetFeatureString(), *TM,
                      true);

  auto *TLI = ST.getTargetLowering();

  for (const auto &Test : Tests) {
    ASSERT_EQ(TLI->isLegalAddImmediate(Test.FixedImm, Test.ScalableImm),
              Test.Result);
  }
}
