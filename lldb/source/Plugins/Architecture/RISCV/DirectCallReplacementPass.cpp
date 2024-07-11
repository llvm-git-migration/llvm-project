//===--- DirectCallReplacementPass.cpp - RISC-V specific pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "Plugins/Architecture/RISCV/DirectCallReplacementPass.h"

#include "lldb/Core/Architecture.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <optional>

using namespace llvm;
using namespace lldb_private;

namespace {
std::string GetValueTypeStr(const llvm::Type *value_ty) {
  assert(value_ty);
  std::string str_type;
  llvm::raw_string_ostream rso(str_type);
  value_ty->print(rso);
  return rso.str();
}

std::string getFunctionName(const llvm::CallInst *ci) {
  return ci->isIndirectCall() ? "indirect"
                              : ci->getCalledFunction()->getName().str();
}
} // namespace

bool DirectCallReplacementPass::canBeReplaced(const llvm::CallInst *ci) {
  assert(ci);
  Log *log = GetLog(LLDBLog::Expressions);

  auto *return_value_ty = ci->getType();
  if (!(return_value_ty->isIntegerTy() || return_value_ty->isVoidTy() ||
        return_value_ty->isPointerTy())) {
    LLDB_LOG(log,
             "DirectCallReplacementPass::{0}() function {1} has unsupported "
             "return type ({2})\n",
             __FUNCTION__, getFunctionName(ci),
             GetValueTypeStr(return_value_ty));
    return false;
  }

  const auto *arg = llvm::find_if_not(ci->args(), [](const auto &arg) {
    const auto *type = arg->getType();
    return type->isIntegerTy() || type->isPointerTy();
  });

  if (arg != ci->arg_end()) {
    LLDB_LOG(log,
             "DirectCallReplacementPass::{0}() argument {1} of {2} function "
             "has unsupported type ({3})\n",
             __FUNCTION__, (*arg)->getName(), getFunctionName(ci),
             GetValueTypeStr((*arg)->getType()));
    return false;
  }
  return true;
}

std::vector<llvm::Value *>
DirectCallReplacementPass::getFunctionArgsAsValues(const llvm::CallInst *ci) {
  assert(ci);
  std::vector<llvm::Value *> args{};
  llvm::transform(ci->args(), std::back_inserter(args),
                  [](const auto &arg) { return arg.get(); });
  return args;
}

std::optional<lldb::addr_t>
DirectCallReplacementPass::getFunctionAddress(const llvm::CallInst *ci) const {
  Log *log = GetLog(LLDBLog::Expressions);

  auto *target = m_exe_ctx.GetTargetPtr();
  const auto &lldb_module_sp = target->GetExecutableModule();
  const auto &symtab = lldb_module_sp->GetSymtab();
  const llvm::StringRef name = ci->getCalledFunction()->getName();

  // eSymbolTypeCode: we try to find function
  // eDebugNo: not a debug symbol
  // eVisibilityExtern: function from extern module
  const auto *symbol = symtab->FindFirstSymbolWithNameAndType(
      ConstString(name), lldb::SymbolType::eSymbolTypeCode,
      Symtab::Debug::eDebugNo, Symtab::Visibility::eVisibilityExtern);
  if (!symbol) {
    LLDB_LOG(log, "DirectCallReplacementPass::{0}() can't find {1} in symtab\n",
             __FUNCTION__, name);
    return std::nullopt;
  }

  lldb::addr_t addr = symbol->GetLoadAddress(target);
  LLDB_LOG(
      log,
      "DirectCallReplacementPass::{0}() found address ({1:x}) of symbol {2}\n",
      __FUNCTION__, addr, name);
  return addr;
}

llvm::CallInst *
DirectCallReplacementPass::getInstReplace(llvm::CallInst *ci) const {
  assert(ci);

  std::optional<lldb::addr_t> addr_or_null = getFunctionAddress(ci);
  if (!addr_or_null.has_value())
    return nullptr;

  lldb::addr_t addr = addr_or_null.value();

  llvm::IRBuilder<> builder(ci);

  std::vector<llvm::Value *> args = getFunctionArgsAsValues(ci);
  llvm::Constant *func_addr = builder.getInt64(addr);
  llvm::PointerType *ptr_func_ty = builder.getPtrTy();
  auto *cast = builder.CreateIntToPtr(func_addr, ptr_func_ty);
  auto *new_inst =
      builder.CreateCall(ci->getFunctionType(), cast, ArrayRef(args));
  return new_inst;
}

DirectCallReplacementPass::DirectCallReplacementPass(
    const ExecutionContext &exe_ctx)
    : FunctionPass(ID), m_exe_ctx{exe_ctx} {}

DirectCallReplacementPass::~DirectCallReplacementPass() = default;

bool DirectCallReplacementPass::runOnFunction(llvm::Function &func) {
  bool has_irreplaceable =
      llvm::any_of(instructions(func), [this](llvm::Instruction &inst) {
        llvm::CallInst *ci = dyn_cast<llvm::CallInst>(&inst);
        if (!ci)
          return false;

        // The function signature does not match the call signature.
        if (!ci->isIndirectCall() && !ci->getCalledFunction())
          return true;

        if (!ci->isIndirectCall() && ci->getCalledFunction()->isIntrinsic())
          return false;

        if (DirectCallReplacementPass::canBeReplaced(ci) &&
            getFunctionAddress(ci).has_value())
          return false;

        return true;
      });

  if (has_irreplaceable) {
    func.getParent()->getOrInsertNamedMetadata(
        Architecture::s_target_incompatibility_marker);
    return false;
  }

  std::vector<std::reference_wrapper<llvm::Instruction>>
      replaceable_function_calls{};
  llvm::copy_if(instructions(func),
                std::back_inserter(replaceable_function_calls),
                [](llvm::Instruction &inst) {
                  llvm::CallInst *ci = dyn_cast<llvm::CallInst>(&inst);
                  if (ci && !ci->isIndirectCall() &&
                      !ci->getCalledFunction()->isIntrinsic())
                    return true;
                  return false;
                });

  if (replaceable_function_calls.empty())
    return false;

  std::vector<std::pair<llvm::CallInst *, llvm::CallInst *>> replaces;
  llvm::transform(replaceable_function_calls, std::back_inserter(replaces),
                  [this](std::reference_wrapper<llvm::Instruction> inst)
                      -> std::pair<llvm::CallInst *, llvm::CallInst *> {
                    llvm::CallInst *ci = cast<llvm::CallInst>(&(inst.get()));
                    llvm::CallInst *new_inst = getInstReplace(ci);
                    return {ci, new_inst};
                  });

  for (auto &&[from, to] : replaces) {
    from->replaceAllUsesWith(to);
    from->eraseFromParent();
  }

  return true;
}

llvm::StringRef DirectCallReplacementPass::getPassName() const {
  return "Transform function calls to calls by address";
}

char DirectCallReplacementPass::ID = 0;

llvm::FunctionPass *
lldb_private::createDirectCallReplacementPass(const ExecutionContext &exe_ctx) {
  return new DirectCallReplacementPass(exe_ctx);
}
