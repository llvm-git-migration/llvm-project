//===-- ExpandVariadicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an optimisation pass for variadic functions. If called from codegen,
// it can serve as the implementation of variadic functions for a given target.
//
// The target-dependent parts are in namespace VariadicABIInfo. Enabling a new
// target means adding a case to VariadicABIInfo::create() along with tests.
//
// The module pass using that information is class ExpandVariadics.
//
// The strategy is:
// 1. Test whether a variadic function is sufficiently simple
// 2. If it was, calls to it can be replaced with calls to a different function
// 3. If it wasn't, try to split it into a simple function and a remainder
// 4. Optionally rewrite the varadic function calling convention as well
//
// This pass considers "sufficiently simple" to mean a variadic function that
// calls into a different function taking a va_list to do the real work. For
// example, libc might implement fprintf as a single basic block calling into
// vfprintf. This pass can then rewrite call to the variadic into some code
// to construct a target-specific value to use for the va_list and a call
// into the non-variadic implementation function. There's a test for that.
//
// Most other variadic functions whose definition is known can be converted into
// that form. Create a new internal function taking a va_list where the original
// took a ... parameter. Move the blocks across. Create a new block containing a
// va_start that calls into the new function. This is nearly target independent.
//
// Where this transform is consistent with the ABI, e.g. AMDGPU or NVPTX, or
// where the ABI can be chosen to align with this transform, the function
// interface can be rewritten along with calls to unknown variadic functions.
//
// The aggregate effect is to unblock other transforms, most critically the
// general purpose inliner. Known calls to variadic functions become zero cost.
//
// This pass does define some target specific information which is partially
// redundant with other parts of the compiler. In particular, the call frame
// it builds must be the exact complement of the va_arg lowering performed
// by clang. The va_list construction is similar to work done by the backend
// for targets that lower variadics there, though distinct in that this pass
// constructs the pieces using alloca instead of relative to stack pointers.
//
// Consistency with clang is primarily tested by emitting va_arg using clang
// then expanding the variadic functions using this pass, followed by trying
// to constant fold the functions to no-ops.
//
// Target specific behaviour is tested in IR - mainly checking that values are
// put into positions in call frames that make sense for that particular target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "expand-variadics"

using namespace llvm;

namespace {
namespace VariadicABIInfo {

// calling convention for passing as valist object, same as it would be in C
// aarch64 uses byval
enum class valistCC { value, pointer, /*byval*/ };

struct Interface {
protected:
  Interface() {}

public:
  virtual ~Interface() {}

  // Most ABIs use a void* or char* for va_list, others can specialise
  virtual Type *vaListType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  // How the vaListType is passed
  virtual valistCC vaListCC() { return valistCC::value; }

  // The valist might need to be stack allocated.
  virtual bool valistOnStack() { return false; }

  virtual void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                                AllocaInst *va_list, Value * /*buffer*/) {
    // Function needs to be implemented if valist is on the stack
    assert(!valistOnStack());
    assert(va_list == nullptr);
  }

  virtual uint32_t minimum_slot_align() = 0;
  virtual uint32_t maximum_slot_align() = 0;

  // Could make these virtual, fair chance that's free since all
  // classes choose not to override them at present

  // All targets currently implemented use a ptr for the valist parameter
  Type *vaListParameterType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  bool VAEndIsNop() { return true; }

  bool VACopyIsMemcpy() { return true; }
};

struct X64SystemV final : public Interface {
  Type *vaListType(LLVMContext &Ctx) override {
    auto I32 = Type::getInt32Ty(Ctx);
    auto Ptr = PointerType::getUnqual(Ctx);
    return ArrayType::get(StructType::get(Ctx, {I32, I32, Ptr, Ptr}), 1);
  }
  valistCC vaListCC() override { return valistCC::pointer; }

  bool valistOnStack() override { return true; }

  void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                        AllocaInst *va_list, Value *voidBuffer) override {
    assert(valistOnStack());
    assert(va_list != nullptr);
    assert(va_list->getAllocatedType() == vaListType(Ctx));

    Type *va_list_ty = vaListType(Ctx);

    Type *I32 = Type::getInt32Ty(Ctx);
    Type *I64 = Type::getInt64Ty(Ctx);

    Value *Idxs[3] = {
        ConstantInt::get(I64, 0),
        ConstantInt::get(I32, 0),
        nullptr,
    };

    Idxs[2] = ConstantInt::get(I32, 0);
    Builder.CreateStore(
        ConstantInt::get(I32, 48),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "gp_offset"));

    Idxs[2] = ConstantInt::get(I32, 1);
    Builder.CreateStore(
        ConstantInt::get(I32, 6 * 8 + 8 * 16),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "fp_offset"));

    Idxs[2] = ConstantInt::get(I32, 2);
    Builder.CreateStore(voidBuffer,
                        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs,
                                                  "overfow_arg_area"));

    Idxs[2] = ConstantInt::get(I32, 3);
    Builder.CreateStore(
        ConstantPointerNull::get(PointerType::getUnqual(Ctx)),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "reg_save_area"));
  }

  // X64 documented behaviour:
  // Slots are at least eight byte aligned and at most 16 byte aligned.
  // If the type needs more than sixteen byte alignment, it still only gets
  // that much alignment on the stack.
  // X64 behaviour in clang:
  // Slots are at least eight byte aligned and at most naturally aligned
  // This matches clang, not the ABI docs.
  uint32_t minimum_slot_align() override { return 8; }
  uint32_t maximum_slot_align() override { return 0; }
};

std::unique_ptr<Interface> create(Module &M) {
  Triple Trip = Triple(M.getTargetTriple());
  const bool IsLinuxABI = Trip.isOSLinux() || Trip.isOSCygMing();

  switch (Trip.getArch()) {

  case Triple::x86: {
    // These seem to all fall out the same, despite getTypeStackAlign
    // implying otherwise.
    if (Trip.isOSDarwin()) {
      struct X86Darwin final : public Interface {
        // X86_32ABIInfo::getTypeStackAlignInBytes is misleading for this.
        // The slotSize(4) implies a minimum alignment
        // The AllowHigherAlign = true means there is no maximum alignment.
        uint32_t minimum_slot_align() override { return 4; }
        uint32_t maximum_slot_align() override { return 0; }
      };

      return std::make_unique<X86Darwin>();
    }
    if (Trip.getOS() == llvm::Triple::Win32) {
      struct X86Windows final : public Interface {
        uint32_t minimum_slot_align() override { return 4; }
        uint32_t maximum_slot_align() override { return 0; }
      };
      return std::make_unique<X86Windows>();
    }

    if (IsLinuxABI) {
      struct X86Linux final : public Interface {
        uint32_t minimum_slot_align() override { return 4; }
        uint32_t maximum_slot_align() override { return 0; }
      };
      return std::make_unique<X86Linux>();
    }
    break;
  }

  case Triple::x86_64: {
    if (Trip.isWindowsMSVCEnvironment() || Trip.isOSWindows()) {
      struct X64Windows final : public Interface {
        uint32_t minimum_slot_align() override { return 8; }
        uint32_t maximum_slot_align() override { return 8; }
      };
      // x64 msvc emit vaarg passes > 8 byte values by pointer
      // however the variadic call instruction created does not, e.g.
      // a <4 x f32> will be passed as itself, not as a pointer or byval.
      // Postponing resolution of that for now.
      return nullptr;
    }

    if (Trip.isOSDarwin()) {
      return std::make_unique<VariadicABIInfo::X64SystemV>();
    }

    if (IsLinuxABI) {
      return std::make_unique<VariadicABIInfo::X64SystemV>();
    }

    break;
  }

  default:
    return nullptr;
  }

  return nullptr;
}

} // namespace VariadicABIInfo

class ExpandVariadics : public ModulePass {
public:
  static char ID;
  std::unique_ptr<VariadicABIInfo::Interface> ABI;

  // todo: drop the boolean
  ExpandVariadics() : ModulePass(ID) {}
  StringRef getPassName() const override { return "Expand variadic functions"; }

  // A predicate in that return nullptr means false
  // Returns the function target to use when inlining on success
  Function *isFunctionInlinable(Module &M, Function *F);

  // Rewrite a call site.
  void ExpandCall(Module &M, CallInst *CB, Function *VarargF, Function *NF);

  // this could be partially target specific
  bool expansionApplicableToFunction(Module &M, Function *F) {
    if (F->isIntrinsic() || !F->isVarArg() ||
        F->hasFnAttribute(Attribute::Naked))
      return false;

    if (F->getCallingConv() != CallingConv::C)
      return false;

    if (GlobalValue::isInterposableLinkage(F->getLinkage()))
      return false;

    for (const Use &U : F->uses()) {
      const auto *CB = dyn_cast<CallBase>(U.getUser());

      if (!CB)
        return false;

      if (CB->isMustTailCall()) {
        return false;
      }

      if (!CB->isCallee(&U) || CB->getFunctionType() != F->getFunctionType()) {
        return false;
      }
    }

    // Branch funnels look like variadic functions but arent:
    //
    // define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) {
    //  musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr @vt1_1,
    //  ptr @vf1_1, ...) ret void
    // }
    //
    // %1 = call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr
    // %obj, i32 1)

    // todo: there should be a reasonable way to check for an intrinsic
    // without inserting a prototype that then needs to be removed
    Function *funnel =
        Intrinsic::getDeclaration(&M, Intrinsic::icall_branch_funnel);
    for (const User *U : funnel->users()) {
      if (auto *I = dyn_cast<CallBase>(U)) {
        if (F == I->getFunction()) {
          return false;
        }
      }
    }
    if (funnel->use_empty())
      funnel->eraseFromParent();

    return true;
  }

  template <Intrinsic::ID ID>
  static BasicBlock::iterator
  skipIfInstructionIsSpecificIntrinsic(BasicBlock::iterator Iter) {
    if (auto *Intrinsic = dyn_cast<IntrinsicInst>(&*Iter))
      if (Intrinsic->getIntrinsicID() == ID)
        Iter++;
    return Iter;
  }

  bool callinstRewritable(CallBase *CB, Function *NF) {
    if (CallInst *CI = dyn_cast<CallInst>(CB))
      if (CI->isMustTailCall())
        return false;

    return true;
  }

  bool runOnFunction(Module &M, Function *F) {
    bool changed = false;

    if (!expansionApplicableToFunction(M, F))
      return false;

    Function *Equivalent = isFunctionInlinable(M, F);

    if (!Equivalent)
      return changed;

    for (User *U : llvm::make_early_inc_range(F->users()))
      if (CallInst *CB = dyn_cast<CallInst>(U)) {
        Value *calledOperand = CB->getCalledOperand();
        if (F == calledOperand) {
          ExpandCall(M, CB, F, Equivalent);
          changed = true;
        }
      }

    return changed;
  }

  bool runOnModule(Module &M) override {
    ABI = VariadicABIInfo::create(M);
    if (!ABI)
      return false;

    bool Changed = false;
    for (Function &F : llvm::make_early_inc_range(M)) {
      Changed |= runOnFunction(M, &F);
    }

    return Changed;
  }
};

Function *ExpandVariadics::isFunctionInlinable(Module &M, Function *F) {
  assert(F->isVarArg());
  assert(expansionApplicableToFunction(M, F));

  if (F->isDeclaration())
    return nullptr;

  // A variadic function is inlinable if it is sufficiently simple.
  // Specifically, if it is a single basic block which is functionally
  // equivalent to packing the variadic arguments into a va_list which is
  // passed to another function. The inlining strategy is to build a va_list
  // in the caller and then call said inner function.

  // Single basic block.
  BasicBlock &BB = F->getEntryBlock();
  if (!isa<ReturnInst>(BB.getTerminator())) {
    return nullptr;
  }

  // Walk the block in order checking for specific instructions, some of them
  // optional.
  BasicBlock::iterator it = BB.begin();

  AllocaInst *alloca = dyn_cast<AllocaInst>(&*it++);
  if (!alloca)
    return nullptr;

  Value *valist_argument = alloca;

  it = skipIfInstructionIsSpecificIntrinsic<Intrinsic::lifetime_start>(it);

  VAStartInst *start = dyn_cast<VAStartInst>(&*it++);
  if (!start || start->getArgList() != valist_argument) {
    return nullptr;
  }

  // The va_list instance is stack allocated
  // The ... replacement is a va_list passed "by value"
  // That involves a load for some ABIs and passing the pointer for others
  Value *valist_trailing_argument = nullptr;
  switch (ABI->vaListCC()) {
  case VariadicABIInfo::valistCC::value: {
    // If it's being passed by value, need a load
    // TODO: Check it's loading the right thing
    auto *load = dyn_cast<LoadInst>(&*it);
    if (!load)
      return nullptr;
    valist_trailing_argument = load;
    it++;
    break;
  }
  case VariadicABIInfo::valistCC::pointer: {
    // If it's being passed by pointer, going to use the alloca directly
    valist_trailing_argument = valist_argument;
    break;
  }
  }

  CallInst *call = dyn_cast<CallInst>(&*it++);
  if (!call)
    return nullptr;

  if (auto *end = dyn_cast<VAEndInst>(&*it)) {
    if (end->getArgList() != valist_argument)
      return nullptr;
    it++;
  } else {
    // Only fail on a missing va_end if it wasn't a no-op
    if (!ABI->VAEndIsNop())
      return nullptr;
  }

  it = skipIfInstructionIsSpecificIntrinsic<Intrinsic::lifetime_end>(it);

  ReturnInst *ret = dyn_cast<ReturnInst>(&*it++);
  if (!ret || it != BB.end())
    return nullptr;

  // The function call is expected to take the fixed arguments then the alloca
  // TODO: Drop the vectors here, iterate over them both together instead.
  SmallVector<Value *> FuncArgs;
  for (Argument &A : F->args())
    FuncArgs.push_back(&A);

  SmallVector<Value *> CallArgs;
  for (Use &A : call->args())
    CallArgs.push_back(A);

  size_t Fixed = FuncArgs.size();
  if (Fixed + 1 != CallArgs.size())
    return nullptr;

  for (size_t i = 0; i < Fixed; i++)
    if (FuncArgs[i] != CallArgs[i])
      return nullptr;

  if (CallArgs[Fixed] != valist_trailing_argument)
    return nullptr;

  // Check the varadic function returns the result of the inner call
  Value *maybeReturnValue = ret->getReturnValue();
  if (call->getType()->isVoidTy()) {
    if (maybeReturnValue != nullptr)
      return nullptr;
  } else {
    if (maybeReturnValue != call)
      return nullptr;
  }

  // All checks passed. Found a va_list taking function we can use.
  return call->getCalledFunction();
}

void ExpandVariadics::ExpandCall(Module &M, CallInst *CB, Function *VarargF,
                                 Function *NF) {
  const DataLayout &DL = M.getDataLayout();

  if (!callinstRewritable(CB, NF)) {
    return;
  }

  // This is something of a problem because the call instructions' idea of the
  // function type doesn't necessarily match reality, before or after this
  // pass
  // Since the plan here is to build a new instruction there is no
  // particular benefit to trying to preserve an incorrect initial type
  // If the types don't match and we aren't changing ABI, leave it alone
  // in case someone is deliberately doing dubious type punning through a
  // varargs
  FunctionType *FuncType = CB->getFunctionType();
  if (FuncType != VarargF->getFunctionType()) {
    return;
  }

  auto &Ctx = CB->getContext();

  struct slotAlignTy {
    uint32_t min;
    uint32_t max;
  };

  slotAlignTy slotAlign;
  slotAlign.min = ABI->minimum_slot_align();
  slotAlign.max = ABI->maximum_slot_align();

  // Align the struct on slotAlign.min to start with
  Align MaxFieldAlign(slotAlign.min ? slotAlign.min : 1);

  // The strategy here is to allocate a call frame containing the variadic
  // arguments laid out such that a target specific va_list can be initialised
  // with it, such that target specific va_arg instructions will correctly
  // iterate over it. Primarily this means getting the alignment right.

  class {
    // The awkward memory layout is to allow access to a contiguous array of
    // types
    enum { N = 4 };
    SmallVector<Type *, N> fieldTypes;
    SmallVector<std::pair<Value *, bool>, N> maybeValueIsByval;

  public:
    void append(Type *t, Value *v, bool isByval) {
      fieldTypes.push_back(t);
      maybeValueIsByval.push_back({v, isByval});
    }

    void padding(LLVMContext &Ctx, uint64_t by) {
      append(ArrayType::get(Type::getInt8Ty(Ctx), by), nullptr, false);
    }

    size_t size() { return fieldTypes.size(); }
    bool empty() { return fieldTypes.empty(); }

    StructType *asStruct(LLVMContext &Ctx, StringRef Name) {
      const bool isPacked = true;
      return StructType::create(Ctx, fieldTypes,
                                (Twine(Name) + ".vararg").str(), isPacked);
    }

    void initialiseStructAlloca(const DataLayout &DL, IRBuilder<> &Builder,
                                AllocaInst *alloced) {

      StructType *VarargsTy = cast<StructType>(alloced->getAllocatedType());

      for (size_t i = 0; i < size(); i++) {
        auto [v, isByval] = maybeValueIsByval[i];
        if (!v)
          continue;

        auto r = Builder.CreateStructGEP(VarargsTy, alloced, i);
        if (isByval) {
          Type *ByValType = fieldTypes[i];
          Builder.CreateMemCpy(r, {}, v, {},
                               DL.getTypeAllocSize(ByValType).getFixedValue());
        } else {
          Builder.CreateStore(v, r);
        }
      }
    }
  } Frame;

  uint64_t CurrentOffset = 0;
  for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E; ++I) {
    Value *ArgVal = CB->getArgOperand(I);
    bool isByVal = CB->paramHasAttr(I, Attribute::ByVal);
    Type *ArgType = isByVal ? CB->getParamByValType(I) : ArgVal->getType();
    Align DataAlign = DL.getABITypeAlign(ArgType);

    uint64_t DataAlignV = DataAlign.value();

    // Currently using 0 as a sentinel to mean ignored
    if (slotAlign.min && DataAlignV < slotAlign.min)
      DataAlignV = slotAlign.min;
    if (slotAlign.max && DataAlignV > slotAlign.max)
      DataAlignV = slotAlign.max;

    DataAlign = Align(DataAlignV);
    MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);

    if (uint64_t Rem = CurrentOffset % DataAlignV) {
      // Inject explicit padding to deal with alignment requirements
      uint64_t Padding = DataAlignV - Rem;
      Frame.padding(Ctx, Padding);
      CurrentOffset += Padding;
    }

    Frame.append(ArgType, ArgVal, isByVal);
    CurrentOffset += DL.getTypeAllocSize(ArgType).getFixedValue();
  }

  if (Frame.empty()) {
    // Not passing anything, hopefully va_arg won't try to dereference it
    // Might be a target specific thing whether one can pass nullptr instead
    // of undef i32
    Frame.append(Type::getInt32Ty(Ctx), nullptr, false);
  }

  Function *CBF = CB->getParent()->getParent();

  StructType *VarargsTy = Frame.asStruct(Ctx, CBF->getName());

  BasicBlock &BB = CBF->getEntryBlock();
  IRBuilder<> Builder(&*BB.getFirstInsertionPt());

  // Clumsy call here is to set a specific alignment on the struct instance
  AllocaInst *alloced =
      Builder.Insert(new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr,
                                    MaxFieldAlign),
                     "vararg_buffer");
  assert(alloced->getAllocatedType() == VarargsTy);

  // Initialise the fields in the struct
  // TODO: Lifetime annotate it and alloca in entry
  // Needs to start life shortly before these copies and end immediately after
  // the new call instruction
  Builder.SetInsertPoint(CB);

  Frame.initialiseStructAlloca(DL, Builder, alloced);

  unsigned NumArgs = FuncType->getNumParams();

  SmallVector<Value *> Args;
  Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

  // Initialise a va_list pointing to that struct and pass it as the last
  // argument
  {
    PointerType *voidptr = PointerType::getUnqual(Ctx);
    Value *voidBuffer =
        Builder.CreatePointerBitCastOrAddrSpaceCast(alloced, voidptr);

    if (ABI->valistOnStack()) {
      assert(ABI->vaListCC() == VariadicABIInfo::valistCC::pointer);
      Type *va_list_ty = ABI->vaListType(Ctx);

      // todo: one va_list alloca per function, also lifetime annotate
      AllocaInst *va_list =
          Builder.CreateAlloca(va_list_ty, nullptr, "va_list");

      ABI->initializeVAList(Ctx, Builder, va_list, voidBuffer);
      Args.push_back(va_list);
    } else {
      assert(ABI->vaListCC() == VariadicABIInfo::valistCC::value);
      Args.push_back(voidBuffer);
    }
  }

  // Attributes excluding any on the vararg arguments
  AttributeList PAL = CB->getAttributes();
  if (!PAL.isEmpty()) {
    SmallVector<AttributeSet, 8> ArgAttrs;
    for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
      ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
    PAL =
        AttributeList::get(Ctx, PAL.getFnAttrs(), PAL.getRetAttrs(), ArgAttrs);
  }

  SmallVector<OperandBundleDef, 1> OpBundles;
  CB->getOperandBundlesAsDefs(OpBundles);

  CallInst *NewCB = CallInst::Create(NF, Args, OpBundles, "", CB);

  CallInst::TailCallKind TCK = cast<CallInst>(CB)->getTailCallKind();
  assert(TCK != CallInst::TCK_MustTail); // guarded at prologue

  // It doesn't get to be a tail call any more
  // might want to guard this with arch, x64 and aarch64 document that
  // varargs can't be tail called anyway
  // Not totally convinced this is necessary but dead store elimination
  // decides to discard the stores to the alloca and pass uninitialised
  // memory along instead when the function is marked tailcall
  if (TCK == CallInst::TCK_Tail) {
    TCK = CallInst::TCK_None;
  }
  NewCB->setTailCallKind(TCK);

  NewCB->setAttributes(PAL);
  NewCB->takeName(CB);
  NewCB->setCallingConv(CB->getCallingConv());
  NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

  if (!CB->use_empty()) // dead branch?
  {
    CB->replaceAllUsesWith(NewCB);
  }
  CB->eraseFromParent();
}

} // namespace

char ExpandVariadics::ID = 0;

INITIALIZE_PASS(ExpandVariadics, DEBUG_TYPE, "Expand variadic functions", false,
                false)

ModulePass *llvm::createExpandVariadicsPass() { return new ExpandVariadics(); }

PreservedAnalyses ExpandVariadicsPass::run(Module &M, ModuleAnalysisManager &) {
  return ExpandVariadics().runOnModule(M) ? PreservedAnalyses::none()
                                          : PreservedAnalyses::all();
}
