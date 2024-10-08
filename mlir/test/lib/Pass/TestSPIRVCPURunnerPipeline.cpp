//===------------------ TestSPIRVCPURunnerPipeline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for use by mlir-spirv-cpu-runner tests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

class TestSPIRVCPURunnerPipelinePass
    : public PassWrapper<TestSPIRVCPURunnerPipelinePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVCPURunnerPipelinePass)

  StringRef getArgument() const final {
    return "test-spirv-cpu-runner-pipeline";
  }
  StringRef getDescription() const final {
    return "Runs a series of passes for lowering SPIR-V-dialect MLIR to "
           "LLVM-dialect MLIR intended for mlir-spirv-cpu-runner.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                    mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::DLTIDialect>();
  }

  TestSPIRVCPURunnerPipelinePass() = default;
  TestSPIRVCPURunnerPipelinePass(const TestSPIRVCPURunnerPipelinePass &) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    PassManager passManager(module->getContext(),
                            module->getName().getStringRef());
    if (failed(applyPassManagerCLOptions(passManager)))
      return signalPassFailure();
    passManager.addPass(createGpuKernelOutliningPass());
    passManager.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

    OpPassManager &nestedPM = passManager.nest<spirv::ModuleOp>();
    nestedPM.addPass(spirv::createSPIRVLowerABIAttributesPass());
    nestedPM.addPass(spirv::createSPIRVUpdateVCEPass());
    passManager.addPass(createLowerHostCodeToLLVMPass());
    passManager.addPass(createConvertSPIRVToLLVMPass());

    if (failed(runPipeline(passManager, module)))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestSPIRVCPURunnerPipelinePass() {
  PassRegistration<TestSPIRVCPURunnerPipelinePass>();
}
} // namespace test
} // namespace mlir
