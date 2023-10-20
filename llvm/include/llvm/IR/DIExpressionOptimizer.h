//===- llvm/IR/DIExpressionOptimizer.h - Constant folding of DIExpressions --*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for functions to constant fold DIExpressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DIEXPRESSIONOPTIMIZER_H
#define LLVM_IR_DIEXPRESSIONOPTIMIZER_H

#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

/// Returns true if the Op is a DW_OP_constu.
bool isConstantVal(uint64_t Op);

/// Returns true if an operation and operand result in a No Op.
bool isNeutralElement(uint64_t Op, uint64_t Val);

/// Try to fold constant math operations and return the result if possible.
std::optional<uint64_t> foldOperationIfPossible(uint64_t Op, uint64_t Operand1,
                                                uint64_t Operand2);

/// Returns true if the two operations are commutative and can be folded.
bool operationsAreFoldableAndCommutative(uint64_t Op1, uint64_t Op2);

/// Consume one operator and its operand(s).
void consumeOneOperator(DIExpressionCursor &Cursor, uint64_t &Loc,
                        const DIExpression::ExprOperand &Op);

/// Reset the Cursor to the beginning of the WorkingOps.
void startFromBeginning(uint64_t &Loc, DIExpressionCursor &Cursor,
                        ArrayRef<uint64_t> WorkingOps);

/// This function will canonicalize:
/// 1. DW_OP_plus_uconst to DW_OP_constu <const-val> DW_OP_plus
/// 2. DW_OP_lit<n> to DW_OP_constu <n>
SmallVector<uint64_t>
canonicalizeDwarfOperations(ArrayRef<uint64_t> WorkingOps);

/// This function will convert:
/// 1. DW_OP_constu <const-val> DW_OP_plus to DW_OP_plus_uconst
/// 2. DW_OP_constu, 0 to DW_OP_lit0
SmallVector<uint64_t> optimizeDwarfOperations(ArrayRef<uint64_t> WorkingOps);

/// {DW_OP_constu, 0, DW_OP_[plus, minus, shl, shr]} -> {}
/// {DW_OP_constu, 1, DW_OP_[mul, div]} -> {}
bool tryFoldNoOpMath(ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
                     DIExpressionCursor &Cursor,
                     SmallVectorImpl<uint64_t> &WorkingOps);

/// {DW_OP_constu, Const1, DW_OP_constu, Const2, DW_OP_[plus,
/// minus, mul, div, shl, shr] -> {DW_OP_constu, Const1 [+, -, *, /, <<, >>]
/// Const2}
bool tryFoldConstants(ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
                      DIExpressionCursor &Cursor,
                      SmallVectorImpl<uint64_t> &WorkingOps);

/// {DW_OP_constu, Const1, DW_OP_[plus, mul], DW_OP_constu, Const2,
/// DW_OP_[plus, mul]} -> {DW_OP_constu, Const1 [+, *] Const2, DW_OP_[plus,
/// mul]}
bool tryFoldCommutativeMath(ArrayRef<DIExpression::ExprOperand> Ops,
                            uint64_t &Loc, DIExpressionCursor &Cursor,
                            SmallVectorImpl<uint64_t> &WorkingOps);

/// {DW_OP_constu, Const1, DW_OP_[plus, mul], DW_OP_LLVM_arg, Arg1,
/// DW_OP_[plus, mul], DW_OP_constu, Const2, DW_OP_[plus, mul]} ->
/// {DW_OP_constu, Const1 [+, *] Const2, DW_OP_[plus, mul], DW_OP_LLVM_arg,
/// Arg1, DW_OP_[plus, mul]}
bool tryFoldCommutativeMathWithArgInBetween(
    ArrayRef<DIExpression::ExprOperand> Ops, uint64_t &Loc,
    DIExpressionCursor &Cursor, SmallVectorImpl<uint64_t> &WorkingOps);

#endif // LLVM_IR_DIEXPRESSIONOPTIMIZER_H
