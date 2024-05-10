//===- DIExpressionOptimizer.cpp - Constant folding of DIExpressions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions to constant fold DIExpressions.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DIExpressionOptimizer.h"
#include "llvm/BinaryFormat/Dwarf.h"

bool isConstantVal(uint64_t Op) { return Op == dwarf::DW_OP_constu; }

bool isNeutralElement(uint64_t Op, uint64_t Val) {
  switch (Op) {
  case dwarf::DW_OP_plus:
  case dwarf::DW_OP_minus:
  case dwarf::DW_OP_shl:
  case dwarf::DW_OP_shr:
    return Val == 0;
  case dwarf::DW_OP_mul:
  case dwarf::DW_OP_div:
    return Val == 1;
  default:
    return false;
  }
}

std::optional<uint64_t> foldOperationIfPossible(uint64_t Op, uint64_t Operand1,
                                                uint64_t Operand2) {
  bool ResultOverflowed;
  switch (Op) {
  case dwarf::DW_OP_plus: {
    auto Result = SaturatingAdd(Operand1, Operand2, &ResultOverflowed);
    if (ResultOverflowed)
      return std::nullopt;
    return Result;
  }
  case dwarf::DW_OP_minus: {
    if (Operand1 < Operand2)
      return std::nullopt;
    return Operand1 - Operand2;
  }
  case dwarf::DW_OP_shl: {
    if ((uint64_t)countl_zero(Operand1) < Operand2)
      return std::nullopt;
    return Operand1 << Operand2;
  }
  case dwarf::DW_OP_shr: {
    if ((uint64_t)countr_zero(Operand1) < Operand2)
      return std::nullopt;
    return Operand1 >> Operand2;
  }
  case dwarf::DW_OP_mul: {
    auto Result = SaturatingMultiply(Operand1, Operand2, &ResultOverflowed);
    if (ResultOverflowed)
      return std::nullopt;
    return Result;
  }
  case dwarf::DW_OP_div: {
    if (Operand2)
      return Operand1 / Operand2;
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

bool operationsAreFoldableAndCommutative(uint64_t Op1, uint64_t Op2) {
  if (Op1 != Op2)
    return false;
  switch (Op1) {
  case dwarf::DW_OP_plus:
  case dwarf::DW_OP_mul:
    return true;
  default:
    return false;
  }
}

void consumeOneOperator(DIExpressionCursor &Cursor, uint64_t &Loc,
                        const DIExpression::ExprOperand &Op) {
  Cursor.consume(1);
  Loc = Loc + Op.getSize();
}

void startFromBeginning(uint64_t &Loc, DIExpressionCursor &Cursor,
                        ArrayRef<uint64_t> WorkingOps) {
  Cursor.assignNewExpr(WorkingOps);
  Loc = 0;
}

SmallVector<uint64_t>
canonicalizeDwarfOperations(ArrayRef<uint64_t> WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;
  while (Loc < WorkingOps.size()) {
    auto Op = Cursor.peek();
    /// Expression has no operations, break.
    if (!Op)
      break;
    auto OpRaw = Op->getOp();
    auto OpArg = Op->getArg(0);

    if (OpRaw >= dwarf::DW_OP_lit0 && OpRaw <= dwarf::DW_OP_lit31) {
      ResultOps.push_back(dwarf::DW_OP_constu);
      ResultOps.push_back(OpRaw - dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    if (OpRaw == dwarf::DW_OP_plus_uconst) {
      ResultOps.push_back(dwarf::DW_OP_constu);
      ResultOps.push_back(OpArg);
      ResultOps.push_back(dwarf::DW_OP_plus);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}

SmallVector<uint64_t> optimizeDwarfOperations(ArrayRef<uint64_t> WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;
  while (Loc < WorkingOps.size()) {
    auto Op1 = Cursor.peek();
    /// Expression has no operations, exit.
    if (!Op1)
      break;
    auto Op1Raw = Op1->getOp();
    auto Op1Arg = Op1->getArg(0);

    if (Op1Raw == dwarf::DW_OP_constu && Op1Arg == 0) {
      ResultOps.push_back(dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    auto Op2 = Cursor.peekNext();
    /// Expression has no more operations, copy into ResultOps and exit.
    if (!Op2) {
      uint64_t PrevLoc = Loc;
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
      break;
    }
    auto Op2Raw = Op2->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op2Raw == dwarf::DW_OP_plus) {
      ResultOps.push_back(dwarf::DW_OP_plus_uconst);
      ResultOps.push_back(Op1Arg);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }
    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.append(WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}

bool tryFoldNoOpMath(ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
                     DIExpressionCursor &Cursor,
                     SmallVectorImpl<uint64_t> &WorkingOps) {
  if (isConstantVal(Ops[0].getOp()) &&
      isNeutralElement(Ops[1].getOp(), Ops[0].getArg(0))) {
    WorkingOps.erase(WorkingOps.begin() + Loc, WorkingOps.begin() + Loc + 3);
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

bool tryFoldConstants(ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
                      DIExpressionCursor &Cursor,
                      SmallVectorImpl<uint64_t> &WorkingOps) {
  if (isConstantVal(Ops[0].getOp()) && isConstantVal(Ops[1].getOp())) {
    auto Result = foldOperationIfPossible(Ops[2].getOp(), Ops[0].getArg(0),
                                          Ops[1].getArg(0));
    if (!Result) {
      consumeOneOperator(Cursor, Loc, Ops[0]);
      return true;
    }
    WorkingOps.erase(WorkingOps.begin() + Loc + 2,
                     WorkingOps.begin() + Loc + 5);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

bool tryFoldCommutativeMath(ArrayRef<DIExpression::ExprOperand> Ops,
                            uint64_t &Loc, DIExpressionCursor &Cursor,
                            SmallVectorImpl<uint64_t> &WorkingOps) {

  if (isConstantVal(Ops[0].getOp()) && isConstantVal(Ops[2].getOp()) &&
      operationsAreFoldableAndCommutative(Ops[1].getOp(), Ops[3].getOp())) {
    auto Result = foldOperationIfPossible(Ops[1].getOp(), Ops[0].getArg(0),
                                          Ops[2].getArg(0));
    if (!Result)
      return false;
    WorkingOps.erase(WorkingOps.begin() + Loc + 3,
                     WorkingOps.begin() + Loc + 6);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}

bool tryFoldCommutativeMathWithArgInBetween(
    ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
    DIExpressionCursor &Cursor, SmallVectorImpl<uint64_t> &WorkingOps) {
  if (isConstantVal(Ops[0].getOp()) &&
      Ops[2].getOp() == dwarf::DW_OP_LLVM_arg &&
      isConstantVal(Ops[4].getOp()) &&
      operationsAreFoldableAndCommutative(Ops[1].getOp(), Ops[3].getOp()) &&
      operationsAreFoldableAndCommutative(Ops[3].getOp(), Ops[5].getOp())) {
    auto Result = foldOperationIfPossible(Ops[1].getOp(), Ops[0].getArg(0),
                                          Ops[4].getArg(0));
    if (!Result)
      return false;
    WorkingOps.erase(WorkingOps.begin() + Loc + 6,
                     WorkingOps.begin() + Loc + 9);
    WorkingOps[Loc] = dwarf::DW_OP_constu;
    WorkingOps[Loc + 1] = *Result;
    startFromBeginning(Loc, Cursor, WorkingOps);
    return true;
  }
  return false;
}
