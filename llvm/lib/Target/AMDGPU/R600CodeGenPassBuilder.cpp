#include "R600CodeGenPassBuilder.h"
#include "R600TargetMachine.h"

using namespace llvm;

R600CodeGenPassBuilder::R600CodeGenPassBuilder(
    R600TargetMachine &TM, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC)
    : CodeGenPassBuilder(TM, Opts, PIC) {}

void R600CodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void R600CodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                           CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error R600CodeGenPassBuilder::addInstSelector(AddMachinePass &) const {
  // TODO: Add instruction selector.
  return Error::success();
}
